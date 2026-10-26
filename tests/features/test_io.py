import tempfile
import unittest
from pathlib import Path

from stillface.features.io import is_finished, write_atomic


class TestAtomicWrite(unittest.TestCase):
    def test_writes_final_file_and_leaves_no_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "out.txt"
            write_atomic(path, lambda tmp_path: tmp_path.write_text("hello"))
            self.assertEqual(path.read_text(), "hello")
            self.assertEqual([p.name for p in path.parent.iterdir()], ["out.txt"])

    def test_failed_writer_leaves_no_final_file_and_no_temp_file(self) -> None:
        def broken(tmp_path: Path) -> None:
            tmp_path.write_text("partial")
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.txt"
            with self.assertRaises(RuntimeError):
                write_atomic(path, broken)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_failed_rewrite_keeps_the_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.txt"
            path.write_text("old")
            with self.assertRaises(RuntimeError):
                write_atomic(path, lambda _: (_ for _ in ()).throw(RuntimeError("x")))
            self.assertEqual(path.read_text(), "old")

    def test_temp_name_keeps_the_extension_for_ffmpeg(self) -> None:
        seen: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            write_atomic(Path(tmp) / "o.mp4", lambda p: (seen.append(p.suffix), p.write_bytes(b"")))
        self.assertEqual(seen, [".mp4"])


class TestIsFinished(unittest.TestCase):
    def test_finished_unless_missing_or_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o.csv"
            self.assertFalse(is_finished(path))
            path.write_text("x")
            self.assertTrue(is_finished(path))
            self.assertFalse(is_finished(path, overwrite=True))


if __name__ == "__main__":
    unittest.main()
