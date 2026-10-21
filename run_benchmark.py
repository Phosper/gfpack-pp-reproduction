#!/usr/bin/env python3
"""Evaluate all 128 official validation cases with the existing inference runner.

No training or tuning. Each case runs in a fresh process and retains all outputs.
Only use the trusted official dataset/checkpoint; pickle is not sandboxed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

EXPECTED_COMMIT = "ff75815e24ec50cd7fa8b2abdec1acfe0b55be95"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(rows, total):
    completed = [row for row in rows if row["status"] == "completed"]
    stages = {}
    for stage in ("raw", "enhanced"):
        valid = [row[stage]["best_official_valid_utilization"] for row in completed
                 if row[stage]["best_official_valid_utilization"] is not None]
        stages[stage] = {
            "official_valid_candidates": sum(row[stage]["official_valid_count"] for row in completed),
            "candidates_in_completed_cases": len(completed) * 512,
            "cases_with_official_valid_candidate": len(valid),
            "cases_without_official_valid_candidate": len(completed) - len(valid),
            "mean_best_utilization_valid_cases_only": sum(valid) / len(valid) if valid else None,
            "mean_best_utilization_all_128_failure_as_zero": sum(valid) / total,
        }
    return {
        "planned_cases": total, "attempted_cases": len(rows), "completed_cases": len(completed),
        "failed_cases": sum(row["status"] == "failed" for row in rows),
        "stages": stages,
        "enhanced_selected_official_valid_and_strict_zero_overlap_cases": sum(
            row["enhanced"]["official_valid_count"] > 0 and
            row["enhanced_selected_geometry"].get("zero_overlap_pass", False) for row in completed),
        "strict_geometry_scope": "Selected best official-valid candidate only; fallback preview excluded from successes",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Existing environment/resources/upstream root")
    parser.add_argument("--output", type=Path, required=True, help="New full-benchmark output directory")
    parser.add_argument("--upstream-model-mode", action="store_true",
                        help="Retain upstream default model/dropout mode during no-grad inference; no training")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    runner = Path(__file__).resolve().with_name("run_inference.py")
    dataset = root / "resources/data/datasets/dataset_dental_vali_128.pkl"
    with dataset.open("rb") as stream:
        ids, actions = pickle.load(stream), pickle.load(stream)
    if len(ids) != 128 or len(actions) != 128:
        raise ValueError("Expected exactly 128 paired validation samples")
    if output.exists():
        raise FileExistsError("Refusing to overwrite a previous benchmark")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root / "upstream", text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError("Unexpected upstream version")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root / "upstream", text=True).strip()
    if dirty:
        raise ValueError("Upstream must be unmodified")
    inputs = [root / "resources/checkpoints/score_model.pth", dataset, runner, Path(__file__).resolve()]
    inputs += sorted((root / "resources/data/polygons").glob("*.txt"))
    inputs += sorted((root / "upstream/src").glob("*.py"))
    inputs += sorted((root / "extensions").glob("*.so"))
    if not list((root / "extensions").glob("*.so")):
        raise ValueError("Compiled extensions missing")
    frozen = {str(path): digest(path) for path in inputs}
    output.mkdir(parents=True)
    plan = {"started_utc": datetime.now(timezone.utc).isoformat(), "upstream_commit": commit,
            "sample_indices": list(range(128)), "candidates": 512, "steps": 128, "seed_per_case": 3407,
            "precision": "FP32", "training": False, "input_sha256": frozen,
            "model_mode": "upstream_default_train_dropout" if args.upstream_model_mode else "eval",
            "official_overlap_area_threshold": 50, "strict_overlap_area_tolerance": 1e-9,
            "failures_retained": True, "case_timeout_seconds": 900}
    save(output / "plan.json", plan)
    rows, started = [], time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
               OMP_NUM_THREADS="16", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    for index in range(128):
        case = output / ("sample_%03d" % index)
        command = [sys.executable, str(runner), "--repo", str(root / "upstream"),
                   "--checkpoint", str(root / "resources/checkpoints/score_model.pth"),
                   "--dataset", str(dataset), "--polygons", str(root / "resources/data/polygons"),
                   "--extensions", str(root / "extensions"), "--sample-index", str(index),
                   "--candidates", "512", "--steps", "128", "--output", str(case)]
        if args.upstream_model_mode:
            command.append("--upstream-model-mode")
        row = {"sample_index": index, "status": "failed"}
        try:
            with (output / ("sample_%03d.log" % index)).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=900)
            row["returncode"] = result.returncode
            if result.returncode == 0:
                meta = json.loads((case / "metadata.json").read_text())
                metrics = json.loads((case / "metrics.json").read_text())
                if meta["status"] != "completed" or meta["sample_index"] != index:
                    raise ValueError("Case metadata mismatch")
                if meta.get("model_mode", "eval") != plan["model_mode"]:
                    raise ValueError("Inference model mode mismatch")
                for path, sha in meta["source_sha256"].items():
                    if path in frozen and sha != frozen[path]:
                        raise ValueError("Frozen input changed")
                row.update(status="completed", elapsed_seconds=meta["elapsed_seconds"],
                           sampler_seconds=meta["sampler_seconds"], enhancement_seconds=meta["enhancement_seconds"],
                           gpu_peak_allocated_bytes=meta["gpu_peak_allocated_bytes"],
                           raw=meta["metric_summary"]["raw"], enhanced=meta["metric_summary"]["enhanced"],
                           enhanced_selected_geometry=metrics["enhanced_preview_independent_geometry"])
        except Exception as exc:
            row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        rows.append(row)
        summary = summarize(rows, 128)
        summary.update(status="running", elapsed_seconds=time.monotonic() - started)
        save(output / "summary.json", summary)
        print("case=%03d status=%s attempted=%d/128 completed=%d" %
              (index, row["status"], len(rows), summary["completed_cases"]), flush=True)
    changed = [str(path) for path in inputs if digest(path) != frozen[str(path)]]
    summary = summarize(rows, 128)
    summary.update(status="completed" if not changed and summary["failed_cases"] == 0 else "completed_with_failures",
                   elapsed_seconds=time.monotonic() - started, changed_inputs=changed,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    save(output / "summary.json", summary)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
