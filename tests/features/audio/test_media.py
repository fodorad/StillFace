import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.features.audio.media import MediaError, decode_audio, probe_audio
from tests.helpers import make_audio, make_video


class TestMedia(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_probe_video_with_audio(self):
        clip = make_audio(self.dir / "a.mp4", seconds=1.0, channels=2, video=True)
        info = probe_audio(clip)
        self.assertEqual((info.codec, info.sample_rate, info.channels), ("aac", 48000, 2))
        self.assertAlmostEqual(info.duration_s, 1.0, delta=0.05)
        self.assertAlmostEqual(info.video_duration_s, 1.0, delta=0.15)

    def test_probe_without_audio_and_broken(self):
        self.assertIsNone(probe_audio(make_video(self.dir / "v.mp4", seconds=0.5, audio=False)))
        bad = self.dir / "bad.mp4"
        bad.write_bytes(b"not a video")
        with self.assertRaises(MediaError):
            probe_audio(bad)
        with self.assertRaises(MediaError):
            decode_audio(self.dir / "v.mp4")

    def test_decode_native_and_resampled_mono(self):
        wav = make_audio(self.dir / "s.wav", "0.5*sin(2*PI*440*t)", 1.0, 48000, channels=2)
        x, rate = decode_audio(wav)
        self.assertEqual((x.shape, rate), ((2, 48000), 48000))
        y, rate = decode_audio(wav, sample_rate=16000, mono=True)
        self.assertEqual(rate, 16000)
        self.assertEqual(y.shape[0], 1)
        self.assertAlmostEqual(y.shape[1], 16000, delta=2)
        spectrum = np.abs(np.fft.rfft(y[0]))
        self.assertAlmostEqual(np.argmax(spectrum) * 16000 / y.shape[1], 440, delta=2)
        self.assertAlmostEqual(float(np.abs(y).max()), 0.5, delta=0.02)

    def test_mono_downmix_is_the_mean(self):
        two = self.dir / "two.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "aevalsrc=0.4*sin(2*PI*200*t)|0:s=16000:d=0.5", "-c:a", "pcm_s16le", str(two)],
            check=True,
        )  # fmt: skip
        y, _ = decode_audio(two, mono=True)
        self.assertAlmostEqual(float(np.abs(y).max()), 0.2, delta=0.01)

    def test_decode_failure(self):
        bad = self.dir / "bad.wav"
        bad.write_bytes(b"RIFF....WAVE")
        with self.assertRaises(MediaError):
            decode_audio(bad)


if __name__ == "__main__":
    unittest.main()
