"""Small replay integration fixture; no real Dental performance claim."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class ReplayTests(unittest.TestCase):
    def test_replay_uses_same_saved_input_and_rejects_modified_source(self):
        import torch
        from replay_paper_enhancement import load_verified_inputs
        from test_paper_guarded_enhancement import rectangle
        # Keep fixtures in this writable checkout; some Windows sandboxes
        # expose a temp path but deny access to newly created subdirectories.
        temp_parent = ROOT
        with tempfile.TemporaryDirectory(prefix="paper-replay-test-", dir=temp_parent) as directory:
            base = Path(directory).resolve()
            self.assertTrue(base.is_relative_to(temp_parent))
            source, extension, output = base / "source", base / "extension", base / "output"
            source.mkdir()
            extension.mkdir()
            # Explicit test double. Production replay hashes and loads the
            # source run's compiled extension, not this Python geometry adapter.
            module = extension / "collision_probe.py"
            module.write_text(
                "from test_paper_guarded_enhancement import GeometryNative\n"
                "_adapter = GeometryNative()\n"
                "paper_repulsion_all = _adapter.paper_repulsion_all\n"
                "paper_gap_xy_all = _adapter.paper_gap_xy_all\n", encoding="utf-8")
            build = extension / "build.json"
            save(build, {"paper_constraints": True, "source_unchanged": True,
                         "upstream_commit": "ff75815e24ec50cd7fa8b2abdec1acfe0b55be95",
                         "extension_sha256": digest(module), "test_fixture_only": True})
            raw = torch.tensor([[[2., 2., 1., 0.], [2.5, 2., 1., 0.]],
                                [[3., 2., 1., 0.], [5., 2., 1., 0.]]], dtype=torch.float32)
            torch.save(raw, source / "predictions_raw.pt")
            save(source / "input_sample.json", {"sample_index": 0, "polygon_ids": [0, 0]})
            save(source / "input_geometry.json", {"official_max_contours": [rectangle()]})
            save(source / "raw_metrics.json", {"samples": [{"sample_index": 0,
                 "planned_candidate_count": 2, "candidates": [
                     {"candidate_index": 0, "native_status": "measured", "native_bounds_xyxy": [2., 2., 3.5, 3.]},
                     {"candidate_index": 1, "native_status": "measured", "native_bounds_xyxy": [3., 2., 6., 3.]},
                 ]}]})
            source_hashes = {p.name: digest(p) for p in source.iterdir()}
            save(source / "metadata.json", {"status": "completed", "changed_inputs": [],
                 "planned_candidates": 2, "sample_index": 0, "profile": "test_fixture_only",
                 "output_sha256": source_hashes,
                 "input_sha256": {str(build): digest(build), str(module): digest(module)}})
            result = subprocess.run([sys.executable, str(ROOT / "replay_paper_enhancement.py"),
                     "--source-run", str(source), "--extension", str(extension),
                     "--output", str(output), "--post-iterations", "3"],
                     cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(comparison["same_post_input"])
            self.assertEqual(set(comparison["policies"]), {"legacy_unit", "guarded_backtracking"})
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["changed_inputs"], [])
            self.assertFalse(metadata["model_executed"])
            for name, expected in source_hashes.items():
                self.assertEqual(digest(source / name), expected)
            for policy in comparison["policies"]:
                for dtype in ("float64", "fp32"):
                    report = json.loads((output / policy / (dtype + "_reference.json")).read_text(encoding="utf-8"))
                    self.assertEqual(report["candidate_count"], 2)
                    self.assertEqual(report["feasible_count"], 2)
                    self.assertEqual(len(report["candidates"]), 2)
            # Same filename with changed geometry must not silently replay.
            with (source / "input_geometry.json").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash-mismatched"):
                load_verified_inputs(source, extension)


if __name__ == "__main__":
    unittest.main()
