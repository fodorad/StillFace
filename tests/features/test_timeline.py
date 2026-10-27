import tempfile
import unittest
from pathlib import Path

from stillface.features.timeline import (
    COLUMNS,
    TimelineRow,
    read_timeline,
    sample_frame_indices,
    write_timeline,
)


def make_row(frame_index: int = 0, **overrides) -> TimelineRow:
    fields = dict(
        session_id="7", camera="window", phase="play", frame_index=frame_index,
        time_s=frame_index / 60, person="mother", gaze_x=0.25, gaze_y=0.5, inout=0.9,
        partner_mass=0.7, gaze_class="BABARA", reason="",
    )  # fmt: skip
    return TimelineRow(**{**fields, **overrides})


class TestTimelineCsv(unittest.TestCase):
    def test_roundtrip_is_lossless_including_empty_gaze_fields(self) -> None:
        rows = [
            make_row(0),
            make_row(2, gaze_x=None, gaze_y=None, inout=None, partner_mass=None,
                     gaze_class="NK", reason="occluded"),
            make_row(4, face_visible=True),
            make_row(6, face_visible=False),
        ]  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "t.csv"
            write_timeline(rows, path)
            self.assertEqual(read_timeline(path), rows)

    def test_header_is_the_documented_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            write_timeline([make_row()], path)
            self.assertEqual(
                path.read_text().splitlines()[0],
                "session_id,camera,phase,frame_index,time_s,person,gaze_x,gaze_y,inout,"
                "partner_mass,gaze_class,reason,face_visible",
            )
        self.assertEqual(len(COLUMNS), 13)

    def test_empty_timeline_still_has_a_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            write_timeline([], path)
            self.assertEqual(read_timeline(path), [])

    def test_unknown_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text("a,b\n1,2\n")
            with self.assertRaises(ValueError):
                read_timeline(path)


class TestSampling(unittest.TestCase):
    def test_60fps_clip_at_30hz_takes_every_second_frame(self) -> None:
        self.assertEqual(sample_frame_indices(7, 60.0, 30.0), [0, 2, 4, 6])

    def test_native_rate_keeps_every_frame(self) -> None:
        self.assertEqual(sample_frame_indices(4, 60.0, 60.0), [0, 1, 2, 3])

    def test_non_integer_ratio_stays_on_the_time_grid(self) -> None:
        indices = sample_frame_indices(60, 60.0, 25.0)
        self.assertEqual(indices[:4], [0, 2, 5, 7])
        self.assertEqual(len(set(indices)), len(indices))

    def test_rate_above_clip_fps_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_frame_indices(10, 30.0, 60.0)

    def test_no_frames(self) -> None:
        self.assertEqual(sample_frame_indices(0, 60.0, 30.0), [])


if __name__ == "__main__":
    unittest.main()
