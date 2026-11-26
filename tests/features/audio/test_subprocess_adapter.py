import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from stillface.features.audio.models import InferenceError, InferenceRequest, Segment
from stillface.features.audio.subprocess_adapter import PACKAGE_ROOT, SubprocessAdapter

REPO = PACKAGE_ROOT


def adapter(**kwargs):
    defaults = dict(
        name="echo",
        python=Path(sys.executable),
        module="tests.features.audio.echo_worker",
        worker_version="1",
        weights_revision="r1",
        pythonpath=(REPO,),
        timeout_s=60.0,
    )
    return SubprocessAdapter(**{**defaults, **kwargs})


class TestSubprocessAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wav = Path(self.tmp.name) / "x.wav"

    def tearDown(self):
        self.tmp.cleanup()

    def request(self, key, **options):
        return InferenceRequest(key, self.wav, 2.0, (Segment(0, 1, "baby", "KCHI"),), options)

    def test_results_and_per_request_error(self):
        out = adapter().run([self.request("1/play"), self.request("1/stillface", fail=True)])
        self.assertEqual(out[0].key, "1/play")
        self.assertEqual(out[0].frames["scores"].names, ("a", "b"))
        self.assertEqual(out[0].segments["seg"].values.tolist(), [[0.5]])
        self.assertEqual(out[0].meta["device"], "cpu")
        self.assertEqual(out[0].artefacts, {"txt": "1/play"})
        self.assertIsInstance(out[1], InferenceError)
        self.assertIn("asked to fail", out[1].message)

    def test_worker_ignores_a_stillface_package_in_the_working_directory(self):
        # A second checkout in the caller's working directory must not shadow the code on
        # the worker's PYTHONPATH (python -m puts the working directory first on sys.path).
        decoy = Path(self.tmp.name) / "decoy"
        for package in ("stillface", "tests"):
            (decoy / package).mkdir(parents=True)
            (decoy / package / "__init__.py").write_text("raise ImportError('decoy')\n")
        here = os.getcwd()
        os.chdir(decoy)
        try:
            out = adapter().run([self.request("1/play")])
        finally:
            os.chdir(here)
        self.assertNotIsInstance(out[0], InferenceError, getattr(out[0], "message", ""))
        self.assertEqual(out[0].key, "1/play")

    def test_relative_wav_paths_resolve_against_the_callers_directory(self):
        self.wav.write_bytes(b"")
        here = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            out = adapter().run([replace(self.request("1/play", read=True), wav=Path("x.wav"))])
        finally:
            os.chdir(here)
        self.assertNotIsInstance(out[0], InferenceError, getattr(out[0], "message", ""))

    def test_crash_keeps_finished_results(self):
        out = adapter().run(
            [self.request("a/b"), self.request("a/c", crash=True), self.request("a/d")]
        )
        self.assertEqual(out[0].key, "a/b")
        for item in out[1:]:
            self.assertIsInstance(item, InferenceError)
            self.assertIn("exited with 3", item.message)

    def test_timeout_and_missing_interpreter(self):
        slow = adapter(options={"sleep": "5"}, timeout_s=0.5).run([self.request("a/b")])
        self.assertIn("timed out", slow[0].message)
        gone = adapter(python=Path(self.tmp.name) / "nope").run([self.request("a/b")])
        self.assertIn("could not start", gone[0].message)

    def test_empty_batch_and_weights_required(self):
        self.assertEqual(adapter().run([]), [])
        with self.assertRaises(ValueError):
            adapter(weights_revision="")


if __name__ == "__main__":
    unittest.main()
