"""Synthetic media for tests, generated with ffmpeg."""

import subprocess
from pathlib import Path


def make_video(
    path: Path, seconds: float = 3.0, fps: int = 60, size: str = "64x48", audio: bool = True
) -> Path:
    """Write a small test-pattern video, optionally with a sine tone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}"]  # fmt: skip
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


def make_noise_video(path: Path, delay_ms: int = 0, seconds: float = 6.0, fps: int = 60) -> Path:
    """Write a video whose audio is seeded noise, started ``delay_ms`` late (for sync tests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={fps}:duration={seconds}",
         "-f", "lavfi", "-i", f"anoisesrc=d={seconds - delay_ms / 1000}:seed=7:sample_rate=48000",
         "-af", f"adelay={delay_ms}|{delay_ms}" if delay_ms else "anull",
         "-ar", "48000", "-pix_fmt", "yuv420p", "-shortest", str(path)],
        check=True,
    )  # fmt: skip
    return path
