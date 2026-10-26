import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stillface.features.gaze.heads import HeadBox
from stillface.features.overlay import (
    NOT_CODABLE_COLOR,
    PERSON_COLORS,
    OverlayItem,
    draw_items,
    render_overlay,
    render_overlay_grid,
)
from tests.helpers import make_video

W, H = 160, 120
BOX = HeadBox(0.25, 0.25, 0.5, 0.6)  # pixels x 40-80, y 30-72


def blank() -> np.ndarray:
    return np.zeros((H, W, 3), np.uint8)


def has_color(frame: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool((frame == np.array(color, np.uint8)).all(axis=-1).any())


class TestDrawItems(unittest.TestCase):
    def test_box_uses_the_person_color(self) -> None:
        frame = blank()
        draw_items(frame, [OverlayItem("mother", BOX, None, "")])
        self.assertTrue(has_color(frame, PERSON_COLORS["mother"]))
        self.assertFalse(has_color(frame, PERSON_COLORS["baby"]))

    def test_persons_have_different_colors(self) -> None:
        self.assertNotEqual(PERSON_COLORS["mother"], PERSON_COLORS["baby"])

    def test_pilot_item_draws_box_and_role_without_gaze(self) -> None:
        frame = blank()
        draw_items(frame, [OverlayItem("baby", BOX, None, "")])
        self.assertTrue(frame.any())
        self.assertFalse(frame[90:, :].any(), "nothing is drawn away from the head")

    def test_gaze_point_adds_marks_beyond_the_box(self) -> None:
        without, with_gaze = blank(), blank()
        draw_items(without, [OverlayItem("baby", BOX, None, "X")])
        draw_items(with_gaze, [OverlayItem("baby", BOX, (0.9, 0.9), "X")])
        self.assertTrue(with_gaze[100:, 130:].any())
        self.assertFalse(without[100:, 130:].any())

    def test_not_codable_is_grey_and_shows_the_reason(self) -> None:
        plain, reason = blank(), blank()
        draw_items(plain, [OverlayItem("mother", BOX, None, "")])
        draw_items(reason, [OverlayItem("mother", None, None, "NK:occluded")])
        self.assertTrue(has_color(reason, NOT_CODABLE_COLOR))
        self.assertFalse(has_color(reason, PERSON_COLORS["mother"]))
        self.assertTrue(plain.any())

    def test_nothing_to_draw(self) -> None:
        frame = blank()
        draw_items(frame, [])
        self.assertFalse(frame.any())


def probe_audio_streams(path: Path) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip


class TestRenderOverlay(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.clip = make_video(self.tmp / "clip.mp4", seconds=1, fps=30, size=f"{W}x{H}")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def read(self, path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames

    def test_overlay_keeps_size_fps_and_frame_count_and_drops_audio(self) -> None:
        dst = self.tmp / "out" / "overlay.mp4"
        render_overlay(self.clip, {0: [OverlayItem("mother", BOX, (0.7, 0.5), "BABARA")]}, dst)

        original, overlay = self.read(self.clip), self.read(dst)
        self.assertEqual(len(overlay), len(original))
        self.assertEqual(overlay[0].shape, original[0].shape)
        cap = cv2.VideoCapture(str(dst))
        self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 30, delta=0.1)
        cap.release()
        self.assertEqual(probe_audio_streams(dst), "")
        self.assertGreater(float(np.abs(overlay[0].astype(int) - original[0]).mean()), 0.5)

    def test_annotations_are_held_until_the_next_sampled_frame(self) -> None:
        dst = self.tmp / "o.mp4"
        items = {
            0: [OverlayItem("mother", BOX, None, "")],
            4: [OverlayItem("mother", HeadBox(0.6, 0.6, 0.9, 0.9), None, "")],
        }
        render_overlay(self.clip, items, dst)
        render_overlay(self.clip, {}, self.tmp / "plain.mp4")
        original, overlay = self.read(self.tmp / "plain.mp4"), self.read(dst)

        def changed(index: int, rows: slice, cols: slice) -> float:
            diff = np.abs(overlay[index].astype(int) - original[index].astype(int))
            return float(diff[rows, cols].mean())

        first_box = (slice(25, 80), slice(35, 85))
        second_box = (slice(65, 115), slice(90, 150))
        noise = 1.0  # re-encoding differs slightly away from the annotations
        self.assertGreater(changed(3, *first_box), changed(3, *second_box) + noise, "held from 0")
        self.assertGreater(changed(5, *second_box), changed(5, *first_box) + noise, "new at 4")

    def test_frames_before_the_first_annotation_are_untouched(self) -> None:
        dst = self.tmp / "o.mp4"
        render_overlay(self.clip, {5: [OverlayItem("baby", BOX, None, "")]}, dst)
        render_overlay(self.clip, {}, self.tmp / "plain.mp4")
        original, overlay = self.read(self.tmp / "plain.mp4"), self.read(dst)
        early = np.abs(overlay[0].astype(int) - original[0].astype(int)).mean()
        self.assertLess(float(early), 1.0, "only re-encoding noise")

    def test_existing_output_is_kept_unless_overwrite(self) -> None:
        dst = self.tmp / "o.mp4"
        dst.write_bytes(b"keep")
        render_overlay(self.clip, {}, dst)
        self.assertEqual(dst.read_bytes(), b"keep")
        render_overlay(self.clip, {}, dst, overwrite=True)
        self.assertNotEqual(dst.read_bytes(), b"keep")

    def test_unreadable_clip_raises_and_leaves_nothing(self) -> None:
        dst = self.tmp / "o.mp4"
        with self.assertRaises(RuntimeError):
            render_overlay(self.tmp / "missing.mp4", {}, dst)
        self.assertEqual(list(self.tmp.glob("o*")), [])


class TestOverlayGrid(unittest.TestCase):
    def test_grid_uses_the_overview_layout_and_blacks_out_missing_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baby = make_video(tmp / "baby.mp4", seconds=1, size="64x48", audio=False)
            window = make_video(tmp / "window.mp4", seconds=1, size="64x48", audio=False)
            dst = tmp / "grid" / "session_play.mp4"

            self.assertTrue(
                render_overlay_grid({"baby": baby, "window": window, "mother": None}, dst)
            )

            cap = cv2.VideoCapture(str(dst))
            self.assertEqual((int(cap.get(3)), int(cap.get(4))), (128, 96))
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertLess(float(frame[:48, :64].mean()), 20, "mother (top-left) is black")
            self.assertGreater(float(frame[48:, :64].mean()), 20, "baby (bottom-left) is shown")

    def test_existing_grid_is_kept_unless_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baby = make_video(tmp / "baby.mp4", seconds=1, size="64x48", audio=False)
            dst = tmp / "g.mp4"
            dst.write_bytes(b"keep")
            self.assertTrue(render_overlay_grid({"baby": baby}, dst))
            self.assertEqual(dst.read_bytes(), b"keep")
            self.assertTrue(render_overlay_grid({"baby": baby}, dst, overwrite=True))
            self.assertNotEqual(dst.read_bytes(), b"keep")

    def test_no_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(render_overlay_grid({}, Path(tmp) / "g.mp4"))


if __name__ == "__main__":
    unittest.main()
