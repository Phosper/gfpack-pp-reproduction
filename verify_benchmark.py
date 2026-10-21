#!/usr/bin/env python3
"""Check completed benchmark coverage, frozen inputs, output hashes and summary."""
import argparse
import json
from pathlib import Path

from run_benchmark import digest, summarize


def verify(root):
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if plan["sample_indices"] != list(range(128)):
        raise ValueError("Unexpected planned coverage")
    rows = summary["rows"]
    if [row["sample_index"] for row in rows] != list(range(128)):
        raise ValueError("Missing, duplicated or out-of-order cases")
    for key, value in summarize(rows, 128).items():
        if summary[key] != value:
            raise ValueError("Summary mismatch: " + key)
    if summary["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("Benchmark is still running")
    for source, expected in plan["input_sha256"].items():
        if digest(Path(source)) != expected:
            raise ValueError("Frozen input changed: " + source)
    verified_outputs = 0
    for row in rows:
        case = root / ("sample_%03d" % row["sample_index"])
        if row["status"] != "completed":
            continue
        metadata = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((case / "metrics.json").read_text(encoding="utf-8"))
        if (metadata["status"] != "completed" or metadata["seed"] != 3407 or
                metadata["sample_index"] != row["sample_index"] or
                metadata["candidates"] != 512 or metadata["steps"] != 128):
            raise ValueError("Case configuration mismatch")
        if metadata.get("model_mode", "eval") != plan.get("model_mode", "eval"):
            raise ValueError("Case inference mode mismatch")
        for stage in ("raw", "enhanced"):
            values = metrics[stage]
            if values["candidate_count"] != 512 or len(values["official_valid"]) != 512:
                raise ValueError("Wrong candidate denominator")
            indices = [i for i, flag in enumerate(values["official_valid"]) if flag]
            best = max(indices, key=lambda i: values["utilization"][i]) if indices else None
            if (len(indices) != values["official_valid_count"] or
                    best != values["best_official_valid_index"]):
                raise ValueError("Best-index/validity mismatch")
            for key in row[stage]:
                if row[stage][key] != values[key]:
                    raise ValueError("Per-case summary mismatch")
        if row["enhanced_selected_geometry"] != metrics["enhanced_preview_independent_geometry"]:
            raise ValueError("Geometry summary mismatch")
        if not metadata["output_sha256"]:
            raise ValueError("Missing output hashes")
        for name, expected in metadata["output_sha256"].items():
            path = (case / name).resolve()
            if case.resolve() not in path.parents or digest(path) != expected:
                raise ValueError("Output path/hash mismatch: " + name)
            verified_outputs += 1
    return {"status": "verified", "covered_cases": len(rows),
            "completed_cases": summary["completed_cases"], "failed_cases": summary["failed_cases"],
            "verified_output_files": verified_outputs, "frozen_input_files": len(plan["input_sha256"]),
            "scope": "Coverage/configuration, full saved output hashes, unchanged inputs, summary recomputation; not a new independent geometry recomputation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.output.resolve())
    print(json.dumps(result, indent=2))
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
