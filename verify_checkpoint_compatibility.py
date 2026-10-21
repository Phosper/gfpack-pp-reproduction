#!/usr/bin/env python3
"""Verify recorded compatibility coverage, paired noise and arithmetic on CPU."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

SAMPLES = (0, 45)
CONFIGS = ((1., 500.), (1., 1000.), (.1, 500.), (.1, 1000.))
TIMES = (.01, .05, .15, .35, .60, .85, 1.)
SIGMAS = (1.1, 3., 10., 100., 1000.)
ABS_TOL, REL_TOL = 2e-6, 5e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, atol=ABS_TOL, rtol=REL_TOL):
    if isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label)
        for a, e in zip(actual, expected):
            close(a, e, label, atol, rtol)
    else:
        require(math.isfinite(actual) and math.isfinite(expected), label + ": nonfinite")
        require(math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratios(error, baseline):
    require(all(x > 0 for x in baseline), "Nonpositive baseline")
    return {"channel_ratio_to_paired_zero_score": [e / z for e, z in zip(error, baseline)],
            "translation_ratio_to_paired_zero_score": sum(error[:2]) / sum(baseline[:2]),
            "rotation_ratio_to_paired_zero_score": sum(error[2:]) / sum(baseline[2:]),
            "all_channel_ratio_to_paired_zero_score": sum(error) / sum(baseline)}


def check_ratios(metrics, error, baseline):
    for key, expected in ratios(error, baseline).items():
        close(metrics[key], expected, key)


def verify(report, script):
    import torch
    require(report["status"] == "completed" and not report["changed_inputs"], "Incomplete/changed inputs")
    frozen = [v for k, v in report["input_sha256"].items() if k.endswith("/check_checkpoint_compatibility.py")]
    require(frozen == [sha(script)], "Local diagnostic script differs from frozen script")
    require(report["sample_indices"] == list(SAMPLES) and report["noise_replicates_per_grid_point"] == 4,
            "Unexpected samples/replicates")
    require(report["fixed_time_grid"] == list(TIMES) and report["matched_physical_sigma_grid"] == list(SIGMAS),
            "Unexpected grids")
    require(not any(report[k] for k in ("training_performed", "sampling_performed", "postprocessing_performed")),
            "Unexpected training/sampling/postprocessing")
    expected = {(s, r, i, g, x) for s, r in CONFIGS for i in SAMPLES
                for g, values in (("fixed_t", TIMES), ("matched_sigma", SIGMAS)) for x in values}
    rows = report["grid"]
    key = lambda x: (x["sigma_min"], x["rotation_feature_scale"], x["sample_index"], x["grid_group"], x["grid_value"])
    require(len(rows) == 96 and {key(x) for x in rows} == expected, "Duplicate/missing grid cells")
    require(report["coverage"] == {"planned_grid_cells": 96, "measured_grid_cells": 96,
        "planned_noisy_placements": 384, "measured_noisy_placements": 384,
        "unmeasured_noisy_placements": 0}, "Incorrect coverage")
    for row in rows:
        require(row["status"] == "measured" and row["planned_replicates"] == row["measured_replicates"] == 4
                and row["active_polygons"] == 48, "Incorrect row coverage/shape")
        s, _, _, group, value = key(row)
        t = value if group == "fixed_t" else math.log(value / s) / math.log(1000. / s)
        close(row["t"], t, "Time grid", atol=1e-12, rtol=1e-12)
        sigma = s * (1000. / s) ** t
        close(row["physical_sigma"], sigma, "Physical sigma")
        close(row["sigma_before_epsilon"], sigma, "Recorded sigma")
        metrics = row["metrics"]
        check_ratios(metrics, metrics["channel_normalized_mse"], metrics["channel_zero_score_mse"])
    batches = report["batch_rng"]
    batch_key = lambda b: (b["sigma_min"], b["rotation_feature_scale"], b["sample_index"])
    require(len(batches) == 8 and {batch_key(b) for b in batches} == {(s, r, i) for s, r in CONFIGS for i in SAMPLES},
            "Duplicate/missing batch records")
    for index in SAMPLES:
        noise = torch.randn(48, 48, 4, generator=torch.Generator(device="cpu").manual_seed(3407 + index), dtype=torch.float32)
        digest = hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest()
        paired = [b for b in batches if b["sample_index"] == index]
        require(all(b["noise_sha256"] == digest for b in paired), "CPU epsilon digest mismatch")
        require(len({json.dumps(b["rng"], sort_keys=True) for b in paired}) == 1, "Recorded RNG pairing mismatch")
        for rotation in (500., 1000.):
            require(len({b["matched_sigma_input_sha256"] for b in paired if b["rotation_feature_scale"] == rotation}) == 1,
                    "Recorded matched-sigma inputs differ")
        for position, (group, value) in enumerate([("fixed_t", x) for x in TIMES] + [("matched_sigma", x) for x in SIGMAS]):
            baseline = noise[position * 4:(position + 1) * 4].square().mean(1).double().mean(0).tolist()
            for row in rows:
                if row["sample_index"] == index and row["grid_group"] == group and row["grid_value"] == value:
                    close(row["metrics"]["channel_zero_score_mse"], baseline, "Regenerated epsilon baseline")
    summaries = []
    for group, values in (("fixed_t", TIMES), ("matched_sigma", SIGMAS)):
        for s, rotation in CONFIGS:
            selected = [x for x in rows if x["grid_group"] == group and x["sigma_min"] == s and x["rotation_feature_scale"] == rotation]
            for index in SAMPLES:
                individual = [x["metrics"] for x in selected if x["sample_index"] == index]
                mean = lambda name: [sum(x[name][c] for x in individual) / len(individual) for c in range(4)]
                batch = next(b for b in batches if batch_key(b) == (s, rotation, index))["unweighted_grid_summary"][group]
                error, baseline = mean("channel_normalized_mse"), mean("channel_zero_score_mse")
                close(batch["channel_normalized_mse"], error, "Batch MSE mean")
                close(batch["channel_zero_score_mse"], baseline, "Batch baseline mean")
                check_ratios(batch, error, baseline)
            error = [sum(x["metrics"]["channel_normalized_mse"][c] for x in selected) for c in range(4)]
            baseline = [sum(x["metrics"]["channel_zero_score_mse"][c] for x in selected) for c in range(4)]
            summaries.append({"grid_group": group, "sigma_min": s, "rotation_feature_scale": rotation,
                              "grid_cells": len(selected), **ratios(error, baseline)})
    endpoints = report["sigma_1000_endpoint_checks"]
    require(len(endpoints) == 4 and {(x["rotation_feature_scale"], x["sample_index"]) for x in endpoints}
            == {(r, i) for r in (500., 1000.) for i in SAMPLES}, "Missing endpoint records")
    for endpoint in endpoints:
        require(endpoint["passed"] and endpoint["physical_sigma"] == 1000. and endpoint["both_t"] == 1.
                and endpoint["rtol"] == 1e-5 and endpoint["atol"] == 1e-7, "Endpoint recorded field mismatch")
        close(endpoint["max_absolute_score_difference"], 0., "Recorded endpoint difference", atol=1e-7, rtol=0.)
    require(report["noise_and_dropout_pairing_verified"] and report["sigma_1000_endpoint_pairing_verified"], "Pairing flags false")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    script = Path(__file__).with_name("check_checkpoint_compatibility.py")
    pooled = verify(report, script)
    for mutation in ("duplicate_grid", "tampered_mse"):
        damaged = copy.deepcopy(report)
        if mutation == "duplicate_grid":
            damaged["grid"][0] = copy.deepcopy(damaged["grid"][1])
        else:
            damaged["grid"][0]["metrics"]["channel_normalized_mse"][0] += 1.
        try:
            verify(damaged, script)
        except ValueError:
            continue
        raise ValueError("Verifier failed its negative control: " + mutation)
    import torch
    result = {"status": "verified", "verification_scope": "CPU noise, recorded coverage/pairing and summary arithmetic",
        "report_file": args.report.name, "report_sha256": sha(args.report), "diagnostic_script_sha256": sha(script),
        "verifier_sha256": sha(__file__), "verifier_pytorch": str(torch.__version__), "coverage": report["coverage"],
        "comparison_tolerance": {"absolute": ABS_TOL, "relative": REL_TOL}, "pooled_summaries": pooled,
        "negative_controls": ["duplicate_grid_rejected", "tampered_mse_rejected"],
        "endpoint_scope": "Four recorded endpoint fields checked; no GPU score was recomputed.",
        "limits": ["384 paired evaluations are not 384 independent layouts.",
            "Model scores, dropout masks and matched inputs were not independently regenerated; their recorded hashes/RNG fields were checked.",
            "Only the local diagnostic script input hash was rechecked; checkpoint/data were not rehashed here.",
            "Two GTs and descriptive grid summaries cannot identify training provenance or packing success."]}
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
