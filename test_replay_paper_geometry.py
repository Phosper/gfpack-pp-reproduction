"""Saved-coordinate integration checks; no model or native extension doubles."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paper_metrics import sample_metrics, aggregate_metrics
from paper_selection import select_saved_result
from replay_paper_geometry import run, verified_source
from run_inference import sha256, write_json
from run_paper_alignment import exact_all


class SelectionTests(unittest.TestCase):
    def report(self, areas):
        # Higher utilization for index 0, but its small overlap must disqualify it.
        refs = [{"bounds_xyxy": [0, 0, 2+i, 1], "total_polygon_area": 2.,
                 "max_pairwise_area": area, "sum_pairwise_area": area} for i, area in enumerate(areas)]
        sample = sample_metrics(0, None, refs, planned_candidate_count=len(refs))
        report = aggregate_metrics([sample])
        return {"enhanced_float64": report, "enhanced_fp32": report}

    def test_selects_lower_utilization_valid_candidate(self):
        result = select_saved_result(self.report([.001, 0.]))
        self.assertEqual(result["candidate_index"], 1)
        self.assertFalse(result["fallback_used"])

    def test_all_invalid_has_no_fallback_utilization(self):
        result = select_saved_result(self.report([.001, 2.]))
        self.assertEqual(result["status"], "no_feasible_candidate")
        self.assertIsNone(result["candidate_index"])
        self.assertIsNone(result["strip_utilization"])

    def test_precision_is_explicit_and_no_hidden_cross_dtype_fallback(self):
        reports = {"enhanced_float64": self.report([0.])["enhanced_float64"],
                   "enhanced_fp32": self.report([.001])["enhanced_fp32"]}
        self.assertEqual(select_saved_result(reports)["status"], "feasible")
        self.assertEqual(select_saved_result(reports, "fp32")["status"], "no_feasible_candidate")
        self.assertEqual(select_saved_result(reports)["storage_dtype"], "float64")

    def test_runner_defaults_and_formal_candidate_sizes(self):
        import argparse
        import run_paper_alignment
        parsed = []
        original_parse = argparse.ArgumentParser.parse_args
        def capture(parser, *args, **kwargs):
            result = original_parse(parser, *args, **kwargs)
            parsed.append(result)
            return result
        class StopBeforeResourceAccess(Exception):
            pass
        with tempfile.TemporaryDirectory(prefix=".geometry-cli-", dir=Path(__file__).parent) as tmp:
            for count in (128, 512):
                argv = ["run_paper_alignment.py", "--root", tmp,
                        "--extension", str(Path(tmp)/"build"),
                        "--output", str(Path(tmp)/"new_run"), "--candidates", str(count)]
                with patch("sys.argv", argv), patch.object(argparse.ArgumentParser, "parse_args", capture), \
                        patch.object(run_paper_alignment, "command_snapshot", side_effect=StopBeforeResourceAccess):
                    with self.assertRaises(StopBeforeResourceAccess):
                        run_paper_alignment.main()
                self.assertEqual(parsed[-1].profile, "paper_settings")
                self.assertEqual(parsed[-1].geometry_backend, "paper_sat")
                self.assertEqual(parsed[-1].post_policy, "legacy_unit")
                self.assertEqual(parsed[-1].result_dtype, "float64")
                self.assertFalse(parsed[-1].no_constraints)


class GeometryReplayTests(unittest.TestCase):
    def fixture(self, source):
        source.mkdir()
        polygons = [[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]]
        ids = [0, 0]
        actions = torch.tensor([[[2., 0., 1., 0.], [2.5, 0., 1., 0.]],
                                [[2., 0., 1., 0.], [4., 0., 1., 0.]]], dtype=torch.float32)
        torch.save(actions, source / "predictions_raw.pt")
        write_json(source / "input_sample.json", {"sample_index": 0, "polygon_ids": ids})
        write_json(source / "input_geometry.json", {"official_max_contours": polygons})
        refs = exact_all(polygons, ids, actions.tolist())
        native = {"candidate_count": 2, "official_valid": [True, True], "utilization": [.5, .5],
                  "bounding_boxes_xyxy": [r["bounds_xyxy"] for r in refs],
                  "reported_overlap_area_percent": [0., 0.]}
        write_json(source / "raw_metrics.json", aggregate_metrics([sample_metrics(0, native, refs)]))
        write_json(source / "metadata.json", {"status": "completed", "changed_inputs": [],
            "sample_index": 0, "planned_candidates": 2, "profile": {"profile": "fixture"},
            "output_sha256": {p.name: sha256(p) for p in source.iterdir()}})
        return actions

    def test_replay_without_native_extension_preserves_sources_and_subset(self):
        with tempfile.TemporaryDirectory(prefix=".geometry-test-", dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            source, out = root / "source", root / "result"
            original = self.fixture(source)
            source_hashes = {p.name: sha256(p) for p in source.iterdir()}
            meta = run(source, out, iterations=5, candidate_limit=1)
            self.assertEqual(meta["status"], "completed")
            self.assertEqual(meta["source_planned_candidates"], 2)
            self.assertEqual(meta["planned_candidates"], 1)
            self.assertTrue(meta["subset_smoke"])
            self.assertFalse(meta["model_executed"])
            self.assertEqual(meta["changed_inputs"], [])
            self.assertEqual(source_hashes, {p.name: sha256(p) for p in source.iterdir()})
            stored = torch.load(out / "predictions_enhanced_fp32.pt", weights_only=True)
            self.assertTrue(torch.equal(stored[:, :, 2:], original[:1, :, 2:]))
            selected = json.loads((out / "selected_result.json").read_text())
            self.assertEqual(selected["candidate_index"], 0)
            post = json.loads((out / "post.json").read_text())
            self.assertIn("gap_substages", post["trace"][0]["candidate_stages"])
            self.assertEqual(post["geometry_backend"], "paper_sat")
            with self.assertRaises(ValueError):
                run(source, out)

    def test_hash_tampering_is_rejected_before_tensor_load(self):
        with tempfile.TemporaryDirectory(prefix=".geometry-test-", dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "source"
            self.fixture(source)
            (source / "predictions_raw.pt").write_bytes(b"invalid changed tensor")
            with self.assertRaisesRegex(ValueError, "hash-mismatched"):
                verified_source(source)


if __name__ == "__main__":
    unittest.main()
