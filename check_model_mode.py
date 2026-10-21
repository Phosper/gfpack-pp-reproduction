#!/usr/bin/env python3
"""Paired no-grad inference mode checks; never update weights or tune the model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", nargs="+", type=int, default=[0, 45, 77])
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample in args.samples:
        for mode in ("eval", "upstream_default"):
            output = args.output / ("sample_%03d_%s" % (sample, mode))
            command = [sys.executable, str(Path(__file__).with_name("run_inference.py")),
                       "--repo", str(root / "upstream"), "--checkpoint", str(root / "resources/checkpoints/score_model.pth"),
                       "--dataset", str(root / "resources/data/datasets/dataset_dental_vali_128.pkl"),
                       "--polygons", str(root / "resources/data/polygons"), "--extensions", str(root / "extensions"),
                       "--sample-index", str(sample), "--candidates", "512", "--steps", "128", "--output", str(output)]
            if mode == "upstream_default":
                command.append("--upstream-model-mode")
            with output.with_suffix(".log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            if result.returncode:
                raise RuntimeError("Mode check failed: " + output.name)
            metadata = json.loads((output / "metadata.json").read_text())
            rows.append({"sample_index": sample, "mode": mode, "model_training_flag": metadata["model_training_flag"],
                         "metrics": metadata["metric_summary"], "sampler_seconds": metadata["sampler_seconds"],
                         "checkpoint_sha256": metadata["source_sha256"][str(root / "resources/checkpoints/score_model.pth")]})
            print(json.dumps(rows[-1]), flush=True)
            (args.output / "summary.json").write_text(json.dumps({"training": False,
                "scope": "Deliberately selected diagnostic samples; not a new full benchmark or causal isolation of dropout RNG",
                "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
