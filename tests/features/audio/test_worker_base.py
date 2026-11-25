import json
import tempfile
import unittest
from pathlib import Path

from stillface.features.audio.models import InferenceRequest, load_result, request_to_dict
from stillface.features.audio.workers.base import error_path, result_path, serve
from tests.features.audio.echo_worker import EchoWorker


class TestServe(unittest.TestCase):
    def test_writes_results_and_errors_in_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            requests = [
                InferenceRequest("1/play", Path("x.wav"), 1.0),
                InferenceRequest("1/reunion", Path("x.wav"), 1.0, options={"fail": True}),
            ]
            request_file = Path(tmp) / "req.json"
            request_file.write_text(
                json.dumps(
                    {
                        "out_dir": str(out),
                        "options": {},
                        "requests": [request_to_dict(r) for r in requests],
                    }
                )
            )
            self.assertEqual(serve(EchoWorker(), [str(request_file)]), 0)
            result = load_result(result_path(out, 0))
            self.assertEqual((result.key, result.meta["device"]), ("1/play", "cpu"))
            self.assertIn("ValueError: asked to fail", error_path(out, 1).read_text())
            self.assertFalse(result_path(out, 1).exists())


if __name__ == "__main__":
    unittest.main()
