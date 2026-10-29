#!/usr/bin/env python3
"""Replay corrected paper geometry on authenticated saved raw candidates.

No CUDA, released model, resampling, or historical native extension is needed.
The baseline retains unit collision steps and the 100-iteration ceiling.
Non-unit steps must be requested explicitly and are labelled numerical probes.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

from run_inference import sha256, utc_now, write_json
from replay_paper_enhancement import original_bbox_min_x, reference_report

REQUIRED = ("input_sample.json", "input_geometry.json", "predictions_raw.pt", "raw_metrics.json")


def verified_source(source):
    source = Path(source).resolve()
    meta_path = source / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "completed" or meta.get("changed_inputs") != []:
        raise ValueError("Source must be completed, with unchanged inputs")
    frozen = {str(meta_path): sha256(meta_path)}
    manifest = meta.get("output_sha256", {})
    names = list(REQUIRED)
    names.extend(n for n in ("predictions_enhanced_float64.pt", "predictions_enhanced_fp32.pt")
                 if n in manifest)
    for name in names:
        path = source / name
        if not path.is_file() or not manifest.get(name) or sha256(path) != manifest[name]:
            raise ValueError("Missing or hash-mismatched source artifact: " + name)
        frozen[str(path)] = manifest[name]
    return meta, frozen


def run(source, out, iterations=100, candidate_limit=None, result_dtype="float64", post_step_size=1.,post_policy=None):
    source, out = Path(source).resolve(), Path(out).resolve()
    if type(iterations) is not int or not 0 <= iterations <= 100:
        raise ValueError("post iterations must be in [0,100]")
    if result_dtype not in ("float64", "fp32"):
        raise ValueError("Result dtype must be float64 or fp32")
    if isinstance(post_step_size, bool) or not math.isfinite(post_step_size) or post_step_size<=0:
        raise ValueError("Post step must be positive and finite")
    policy = post_policy or ("legacy_unit" if post_step_size==1. else "paper_fixed_step")
    if policy not in ("legacy_unit","paper_fixed_step","paper_potential_descent"):
        raise ValueError("Unknown reference replay policy")
    if out.exists() or source == out or source in out.parents or out in source.parents:
        raise ValueError("Output must be new and outside the source tree")
    original, frozen = verified_source(source)
    import numpy as np
    import torch
    import shapely
    from paper_sat_geometry import PaperSATGeometry
    from paper_enhancement import enhance_chunked

    sample = json.loads((source / "input_sample.json").read_text(encoding="utf-8"))
    geometry = json.loads((source / "input_geometry.json").read_text(encoding="utf-8"))
    metrics = json.loads((source / "raw_metrics.json").read_text(encoding="utf-8"))
    ids, polygons = sample["polygon_ids"], geometry["official_max_contours"]
    count = original["planned_candidates"]
    raw = torch.load(source / "predictions_raw.pt", map_location="cpu", weights_only=True)
    if (type(count) is not int or count < 1 or not ids or
            not all(type(i) is int and 0 <= i < len(polygons) for i in ids) or
            not isinstance(raw, torch.Tensor) or raw.dtype not in (torch.float32, torch.float64) or
            raw.shape != (count, len(ids), 4) or not torch.isfinite(raw).all() or
            sample["sample_index"] != original["sample_index"]):
        raise ValueError("Input geometry, tensor and source denominator disagree")
    used = count if candidate_limit is None else candidate_limit
    if type(used) is not int or not 1 <= used <= count:
        raise ValueError("candidate-limit must be between 1 and the original candidate count")
    min_x = original_bbox_min_x(metrics, sample["sample_index"], count)
    actions = raw[:used].numpy().astype(np.float64, copy=True)
    actions[:, :, 0] -= np.asarray(min_x[:used])[:, None]
    for name in ("replay_paper_geometry.py", "replay_paper_enhancement.py", "paper_enhancement.py",
                 "paper_sat_geometry.py", "paper_potential_enhancement.py", "run_paper_alignment.py", "paper_metrics.py", "run_inference.py"):
        path = Path(__file__).resolve().with_name(name)
        frozen[str(path)] = sha256(path)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    meta = {"status": "running", "started_utc": utc_now(), "source_run": str(source),
            "sample_index": sample["sample_index"], "source_profile": original.get("profile"),
            "source_geometry_backend": original.get("geometry_backend", "legacy_native"),
            "source_planned_candidates": count, "planned_candidates": used,
            "candidate_indices": list(range(used)), "subset_smoke": used < count,
            "geometry_backend": "paper_sat", "post_policy": policy,
            "post_step_size": float(post_step_size),
            "primary_result_dtype": result_dtype,
            "post_iterations": iterations, "input_sha256": frozen,
            "model_executed": False, "paper_full_reproduction": False,
            "post_input_gauge": "same saved native raw bbox min_x; no y reset",
            "scope": "same-input postprocessing diagnostic, not paper benchmark validation",
            "source_manifest_scope": "self-consistency of supplied manifest; not external authenticity",
            "software": {"python": sys.version, "pytorch": str(torch.__version__),
                         "shapely": shapely.__version__, "geos": shapely.geos_version_string}}
    try:
        before = reference_report(polygons, ids, actions)
        write_json(out / "post_input_reference.json", before)
        final, post = enhance_chunked(PaperSATGeometry(), polygons, ids, actions, iterations,
                              policy=policy, trace_candidates=True,post_step_size=post_step_size)
        write_json(out / "post.json", post)
        reports = {}
        for label, dtype in (("float64", torch.float64), ("fp32", torch.float32)):
            tensor = torch.tensor(final, dtype=dtype)
            torch.save(tensor, out / ("predictions_enhanced_" + label + ".pt"))
            report = reference_report(polygons, ids, tensor.double().numpy())
            write_json(out / (label + "_reference.json"), report)
            reports[label] = {k: v for k, v in report.items() if k != "candidates"}
        saved = reports[result_dtype]
        write_json(out / "selected_result.json", {
            "status": "feasible" if saved["feasible_count"] else "no_feasible_candidate",
            "candidate_index": saved["best_feasible_candidate_index"],
            "strip_utilization": saved["best_feasible_strip_utilization"],
            "source_dtype": result_dtype, "criterion": "reference_strict_fixed_strip",
            "feasible_counts_by_saved_dtype": {k: v["feasible_count"] for k, v in reports.items()},
            "candidate_denominator": used, "source_candidate_denominator": count,
            "fallback_used": False, "semantics": saved["semantics"]})
        historical = {}
        for label in ("float64", "fp32"):
            name = "predictions_enhanced_" + label + ".pt"
            if name not in original.get("output_sha256", {}):
                continue
            tensor = torch.load(source / name, map_location="cpu", weights_only=True)
            expected_dtype = torch.float64 if label == "float64" else torch.float32
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != expected_dtype or
                    tensor.shape != raw.shape or not torch.isfinite(tensor).all()):
                raise ValueError("Invalid saved historical tensor: " + name)
            report = reference_report(polygons, ids, tensor[:used].double().numpy())
            write_json(out / ("historical_" + label + "_reference.json"), report)
            historical[label] = {k: v for k, v in report.items() if k != "candidates"}
        write_json(out / "comparison.json", {
            "same_raw_candidates": True, "candidate_indices": list(range(used)),
            "before": {k: v for k, v in before.items() if k != "candidates"},
            "historical_saved_results": historical, "corrected_geometry": reports,
            "historical_results_available": bool(historical),
            "historical_post_policy": original.get("post_policy", "legacy_unit"),
            "note": "Historical tensors are remeasured, not recomputed; check their saved policy/configuration."})
        meta["status"] = "completed"
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        meta["elapsed_seconds"] = time.perf_counter()-start
        meta["finished_utc"] = utc_now()
        meta["changed_inputs"] = [p for p, digest in frozen.items()
                                  if not Path(p).is_file() or sha256(p) != digest]
        if meta["changed_inputs"]:
            meta["status"] = "failed"
        meta["output_sha256"] = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()}
        write_json(out / "metadata.json", meta)
    if meta["status"] != "completed":
        raise RuntimeError("Frozen inputs changed during replay")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--post-iterations", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, help="Explicit first-N smoke subset; never a full batch claim")
    parser.add_argument("--result-dtype", choices=("float64", "fp32"), default="float64")
    parser.add_argument("--post-step-size",type=float,default=1.,help="Explicit numerical probe; paper does not specify this step")
    parser.add_argument("--post-policy",choices=("legacy_unit","paper_fixed_step","paper_potential_descent"))
    args = parser.parse_args()
    result = run(args.source_run, args.output, args.post_iterations, args.candidate_limit, args.result_dtype,args.post_step_size,args.post_policy)
    print(json.dumps({k: result[k] for k in ("status", "sample_index", "planned_candidates",
                     "source_planned_candidates", "subset_smoke", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
