import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.cli import main
from stillface.config import DatasetPaths
from stillface.features.audio.asr import (
    builder_for,
    flag_words,
    gated_request,
    inside,
    run_asr_flags,
    utterance_request,
    utterances,
)
from stillface.features.audio.config import AudioConfig, ModelConfig, load_audio_config
from stillface.features.audio.infer import full_wav_request, raw_path
from stillface.features.audio.media import write_wav
from stillface.features.audio.models import InferenceResult, Segment, SegmentScores, save_result
from stillface.features.audio.prepare import wav_path
from stillface.features.audio.provenance import write_sidecar
from stillface.features.audio.textgrid import Interval, parse_textgrid, shift, write_textgrid
from stillface.features.audio.workers import crisperwhisper as cw
from stillface.features.audio.workers import mfa

WORDS = (
    Segment(0.5, 0.9, "mother", "szia"),
    Segment(1.0, 1.4, "mother", "baba"),
    Segment(3.0, 3.2, "mother", "hm"),
)


class TestTextGrid(unittest.TestCase):
    def test_round_trip_with_gaps_and_quotes(self):
        tiers = {
            "words": [Interval(0.5, 0.9, 'a"b'), Interval(1.0, 1.4, "baba")],
            "phones": [Interval(0.5, 0.6, "s")],
        }
        text = write_textgrid(tiers, 2.0)
        back = parse_textgrid(text)
        self.assertEqual([i.label for i in back["words"]], ["", 'a"b', "", "baba", ""])
        self.assertEqual(back["words"][1], Interval(0.5, 0.9, 'a"b'))
        self.assertEqual(back["phones"][-1].end_s, 2.0)
        self.assertEqual(shift([Interval(0, 1, "x")], 2.0), [Interval(2.0, 3.0, "x")])


class TestCrisperWhisper(unittest.TestCase):
    def test_postprocess(self):
        words = {
            "verbatim": [(0.5, 0.9, " szia", 0.9), (1.0, 9.0, "öö", 1.2)],
            "intended": [(0.5, 0.9, "szia", 0.8)],
        }
        result = cw.postprocess({"verbatim": "szia öö", "intended": "szia"}, words, 2.0)
        verbatim = result.segments["verbatim_words"]
        self.assertEqual([s.label for s in verbatim.segments], ["szia", "öö"])
        self.assertEqual(verbatim.segments[1].end_s, 2.0)
        self.assertEqual(verbatim.values[1, 0], 1.0)
        doc = json.loads(result.artefacts["transcript.json"])
        self.assertEqual(doc["intended"]["text"], "szia")
        empty = cw.postprocess({}, {}, 1.0)
        self.assertEqual(empty.segments["intended_words"].values.shape, (0, 1))

    def test_infer_with_fake_forward(self):
        class Fake(cw.CrisperWhisperWorker):
            def forward(self, wav):
                return {"verbatim": "a", "intended": "a"}, {
                    "verbatim": [(0, 1, "a", 0.5)],
                    "intended": [],
                }

        from stillface.features.audio.models import InferenceRequest

        result = Fake().infer(InferenceRequest("1/play/mother", Path("x.wav"), 2.0))
        self.assertEqual(result.key, "1/play/mother")


class TestMfa(unittest.TestCase):
    def test_merge_and_oov(self):
        grid = {
            "words": [
                Interval(0.0, 0.1, ""),
                Interval(0.1, 0.4, "szia"),
                Interval(0.4, 0.6, "<unk>"),
            ],
            "phones": [Interval(0.1, 0.2, "s"), Interval(0.2, 0.4, "i"), Interval(0.4, 0.6, "spn")],
        }
        tiers, oov = mfa.merge_alignments(
            [(Segment(1.0, 2.0, "mother", "szia cuncimókus"), grid)], 10.0
        )
        self.assertEqual([i.label for i in tiers["words"]], ["szia", "<unk>"])
        self.assertAlmostEqual(tiers["phones"][0].start_s, 1.1)
        self.assertEqual(oov, ["cuncimókus"])
        _, short = mfa.merge_alignments(
            [(Segment(0, 1, "mother", "a b"), {"words": [Interval(0, 1, "a")]})], 5.0
        )
        self.assertEqual(short, ["b"])
        scores = mfa.to_scores(tiers)
        self.assertEqual(scores["phones"].values.shape, (3, 1))

    def test_infer_with_fake_forward(self):
        class Fake(mfa.MfaWorker):
            def forward(self, request):
                return [(s, {"words": [Interval(0.0, 0.3, s.label)]}) for s in request.segments]

        from stillface.features.audio.models import InferenceRequest

        request = InferenceRequest(
            "1/play/mother", Path("x.wav"), 3.0, (Segment(1.0, 2.0, "mother", "szia"),)
        )
        result = Fake().infer(request)
        self.assertIn('text = "szia"', result.artefacts["TextGrid"])
        self.assertEqual(result.meta["oov"], [])


class TestAsrStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.paths = DatasetPaths(root / "raw", root / "out")
        write_wav(wav_path(self.paths, "1", "play"), np.zeros(16000 * 4), 16000)
        track = wav_path(self.paths, "1", "play", "mother")
        write_wav(track, np.zeros(16000 * 4), 16000)
        write_sidecar(track, {"intervals": [[0.2, 1.7]], "duration_s": 4.0})

    def tearDown(self):
        self.tmp.cleanup()

    def test_helpers(self):
        self.assertTrue(inside(0.5, [(0.2, 1.7)]))
        self.assertFalse(inside(1.7, [(0.2, 1.7)]))
        found = utterances(WORDS, [(0.2, 1.7), (5, 6)])
        self.assertEqual(found, [Segment(0.2, 1.7, "mother", "szia baba")])
        flags = flag_words(WORDS, np.array([[0.9], [0.8], [0.4]]), [(0.2, 1.7)])
        self.assertEqual([f["outside_mother"] for f in flags], [False, False, True])

    def test_builders(self):
        config = load_audio_config()
        self.assertEqual(gated_request("mother")(self.paths, "1", "play").key, "1/play/mother")
        self.assertIsNone(gated_request("baby")(self.paths, "1", "play"))
        self.assertIsNotNone(
            builder_for(config.models["crisperwhisper"], config)(self.paths, "1", "play")
        )
        self.assertIs(builder_for(ModelConfig(), AudioConfig()), full_wav_request)
        aligner = builder_for(config.models["mfa"], config)
        self.assertIsNone(aligner(self.paths, "1", "play"))
        self.assertIsNotNone(builder_for(config.models["egemaps"], config))
        self.store_transcript()
        request = utterance_request()(self.paths, "1", "play")
        self.assertEqual([s.label for s in request.segments], ["szia baba"])
        with self.assertRaises(ValueError):
            ModelConfig(segment_source="asr:")

    def store_transcript(self):
        result = InferenceResult(
            "1/play/mother",
            segments={
                "verbatim_words": SegmentScores(WORDS, ("probability",), [[0.9], [0.8], [0.4]]),
                "intended_words": SegmentScores(WORDS[:2], ("probability",), [[0.9], [0.8]]),
            },
        )
        out = raw_path(self.paths, "crisperwhisper", "1/play/mother")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_result(result, out)

    def test_flags_written(self):
        self.store_transcript()
        aligned = InferenceResult("1/play/mother", meta={"oov": ["cuncimókus"]})
        out = raw_path(self.paths, "mfa", "1/play/mother")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_result(aligned, out)
        shares = run_asr_flags(self.paths, "1", AudioConfig(), ("play", "reunion"))
        doc = json.loads((self.paths.audio_session_dir("1") / "asr" / "play.json").read_text())
        self.assertAlmostEqual(shares["play"], 1 / 3)
        self.assertEqual(doc["verbatim"]["text"], "szia baba hm")
        self.assertEqual(doc["oov"], ["cuncimókus"])
        self.assertEqual(len(doc["intended"]["words"]), 2)
        code = main(
            [
                "--raw-dir",
                self.tmp.name,
                "--processed-dir",
                str(Path(self.tmp.name) / "out"),
                "audio",
                "asr",
                "--session-id",
                "1",
            ]
        )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
