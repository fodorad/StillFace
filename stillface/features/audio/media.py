"""Audio probing and decoding through the ffmpeg command line tools.

ffmpeg is already required by the data preparation and installed in CI, so the audio
pipeline decodes and resamples with it instead of a torch audio stack. Decoded audio is
returned as float32 samples in ``[-1, 1]`` with shape ``(channels, samples)``.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RESAMPLER: str = "resampler=swr:filter_size=64:phase_shift=10:linear_interp=1:cutoff=0.97"
"""Options of ffmpeg's built-in ``aresample`` filter used for every sample-rate conversion.

The built-in resampler with a long windowed-sinc filter is used because soxr is not part of
every ffmpeg build.
"""


class MediaError(RuntimeError):
    """A media file cannot be probed or decoded."""


@dataclass(frozen=True)
class AudioStreamInfo:
    """Properties of the first audio stream of a media file.

    Attributes:
        codec: Codec name, e.g. ``aac``.
        sample_rate: Sample rate in Hz.
        channels: Number of channels.
        duration_s: Audio stream duration in seconds.
        video_duration_s: Duration of the first video stream, None if there is none.
    """

    codec: str
    sample_rate: int
    channels: int
    duration_s: float
    video_duration_s: float | None


def _stream_duration(stream: dict, fallback: float) -> float:
    try:
        return float(stream["duration"])
    except (KeyError, ValueError):
        return fallback


def probe_audio(path: Path) -> AudioStreamInfo | None:
    """Probe the audio stream of a media file.

    Args:
        path: Media file.

    Returns:
        The stream properties, None if the file has no audio stream.

    Raises:
        MediaError: If ffprobe cannot read the file.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed on {path.name}: {result.stderr.strip()}")
    data = json.loads(result.stdout or "{}")
    container = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    streams = data.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    video = [s for s in streams if s.get("codec_type") == "video"]
    if not audio:
        return None
    stream = audio[0]
    return AudioStreamInfo(
        codec=str(stream.get("codec_name", "")),
        sample_rate=int(stream.get("sample_rate", 0)),
        channels=int(stream.get("channels", 0)),
        duration_s=_stream_duration(stream, container),
        video_duration_s=_stream_duration(video[0], container) if video else None,
    )


def decode_audio(
    path: Path, sample_rate: int | None = None, mono: bool = False
) -> tuple[np.ndarray, int]:
    """Decode the first audio stream to float samples.

    Args:
        path: Media file.
        sample_rate: Output rate; the native rate if None. Conversion uses :data:`RESAMPLER`.
        mono: Downmix to one channel by the mean of the channels.

    Returns:
        ``(samples, rate)`` with samples of shape ``(channels, n)``.

    Raises:
        MediaError: If the file has no audio stream or ffmpeg fails.
    """
    info = probe_audio(path)
    if info is None:
        raise MediaError(f"No audio stream in {path.name}")
    rate = sample_rate or info.sample_rate
    filters = []
    if mono and info.channels > 1:
        weight = 1.0 / info.channels
        filters.append("pan=mono|c0=" + "+".join(f"{weight}*c{i}" for i in range(info.channels)))
    if rate != info.sample_rate:
        filters.append(f"aresample={rate}:{RESAMPLER}")
    channels = 1 if mono else info.channels
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn"]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-ac", str(channels), "-ar", str(rate), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise MediaError(f"ffmpeg failed on {path.name}: {result.stderr.decode().strip()}")
    samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32)
    return samples.reshape(-1, channels).T.copy(), rate
