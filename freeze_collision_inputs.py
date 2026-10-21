#!/usr/bin/env python3
"""Freeze matched saved candidates for the 16-case collision development set.

Requires the completed, verified upstream-default-mode 128-case baseline and
its diagnosis. No inference, CUDA, native extensions, or source changes occur.
Only trusted tensor files are loaded, with weights_only=True on the CPU.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

from diagnose_results import selection
from run_benchmark import EXPECTED_COMMIT, digest, summarize


FAILED_SAMPLES = [25, 45, 46, 47, 54, 62, 99, 115]
TOTAL_SAMPLES = 128
REGRESSION_COUNT = 8
CASE_FILES = (
    "input_sample.json", "input_geometry.json", "metrics.json",
    "predictions_raw.pt", "predictions_enhanced.pt", "gt_input.pt",
    "gt_enhanced.pt",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_digest(value):
    """Hash a value using UTF-8, sorted keys, and compact JSON separators."""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    def reject_nonfinite(value):
        raise ValueError("Nonfinite JSON number: " + value)
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=reject_nonfinite)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def equidistant_indices(population_size, count):
    """Include both endpoints; round each interior rational index down.

    For 120 successes and 8 selections this returns
    [0, 17, 34, 51, 68, 85, 102, 119], with no floating-point rounding.
    """
    if count < 2 or population_size < count:
        raise ValueError("Equidistant selection needs population >= count >= 2")
    return [i * (population_size - 1) // (count - 1) for i in range(count)]


def require_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Expected integer " + name)
    return value


def require_hash(value, name):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("Invalid SHA-256 for " + name)


def source_inventory(recorded):
    """Retain all input hashes without disclosing absolute account paths.

    Known project suffixes stay readable. Unknown sources get an opaque
    numbered relative name; source_key_sha256 binds each entry to the exact
    key in the private original plan without copying that key.
    """
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Missing frozen baseline input hashes")
    inventory, used = [], set()
    for number, (source, sha) in enumerate(sorted(recorded.items())):
        require_hash(sha, "baseline input")
        if not isinstance(source, str):
            raise ValueError("Invalid baseline input key")
        parts = PurePosixPath(source.replace("\\", "/")).parts
        relative = None
        for anchor in ("resources", "upstream", "extensions"):
            if anchor in parts:
                suffix = parts[parts.index(anchor):]
                if ".." not in suffix:
                    relative = PurePosixPath(*suffix).as_posix()
                    break
        if parts and parts[-1] in ("run_inference.py", "run_benchmark.py"):
            relative = "runner/" + parts[-1]
        if relative is None or relative in used:
            relative = "other_sources/source_%04d" % number
        used.add(relative)
        inventory.append({"path": relative, "sha256": sha,
                          "source_key_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()})
    return inventory


def indexed_rows(value, name):
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != TOTAL_SAMPLES:
        raise ValueError(name + " must contain 128 rows")
    indices = [require_int(row.get("sample_index"), name + " sample_index") for row in rows]
    if sorted(indices) != list(range(TOTAL_SAMPLES)):
        raise ValueError(name + " has incomplete or duplicated coverage")
    return {row["sample_index"]: row for row in rows}


def validate_baseline(results, diagnosis_path):
    files = {name: results / (name + ".json") for name in ("plan", "summary", "verification")}
    plan, summary, verification = (read_json(files[name]) for name in files)
    diagnosis = read_json(diagnosis_path)
    expected = {"sample_indices": list(range(TOTAL_SAMPLES)), "candidates": 512,
                "steps": 128, "seed_per_case": 3407, "precision": "FP32",
                "training": False, "model_mode": "upstream_default_train_dropout",
                "upstream_commit": EXPECTED_COMMIT,
                "official_overlap_area_threshold": 50,
                "strict_overlap_area_tolerance": 1e-9}
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError("Unexpected baseline configuration: " + key)
    if (summary.get("status") != "completed" or summary.get("completed_cases") != TOTAL_SAMPLES
            or summary.get("failed_cases") != 0 or summary.get("changed_inputs") != []):
        raise ValueError("Expected a completed, unchanged 128-case baseline")
    if (verification.get("status") != "verified" or verification.get("covered_cases") != TOTAL_SAMPLES
            or verification.get("completed_cases") != TOTAL_SAMPLES or verification.get("failed_cases") != 0):
        raise ValueError("Baseline verification is missing or incomplete")
    rows = indexed_rows(summary, "summary")
    for key, value in summarize(summary["rows"], TOTAL_SAMPLES).items():
        if summary.get(key) != value:
            raise ValueError("Baseline summary recomputation mismatch: " + key)
    if diagnosis.get("baseline_summary_sha256") != digest(files["summary"]):
        raise ValueError("Diagnosis belongs to another baseline summary")
    audits = indexed_rows(diagnosis, "diagnosis")
    failed = sorted(index for index, row in rows.items() if row["enhanced"]["official_valid_count"] == 0)
    success = sorted(set(rows) - set(failed))
    if failed != FAILED_SAMPLES or len(success) != 120:
        raise ValueError("Expected the planned 8 failures and 120 official-valid successes")
    if (diagnosis.get("official_valid_group", {}).get("cases") != 120
            or diagnosis.get("no_official_valid_group", {}).get("cases") != 8):
        raise ValueError("Diagnosis group counts disagree with the upstream-mode baseline")
    for index, row in rows.items():
        if row.get("status") != "completed":
            raise ValueError("Baseline contains an incomplete case")
        if audits[index].get("official_valid_candidate_count") != row["enhanced"]["official_valid_count"]:
            raise ValueError("Diagnosis validity count disagrees for sample %03d" % index)
    inventory = source_inventory(plan.get("input_sha256"))
    positions = equidistant_indices(len(success), REGRESSION_COUNT)
    return {"plan": plan, "summary": summary, "verification": verification,
            "diagnosis": diagnosis, "rows": rows, "audits": audits,
            "inventory": inventory, "positions": positions,
            "successes": [success[position] for position in positions],
            "files": {**files, "diagnosis": diagnosis_path},
            "configuration": expected}


def validate_metrics(stage, count, name):
    if stage.get("candidate_count") != count:
        raise ValueError("Wrong candidate count in " + name)
    fields = ("official_valid", "utilization", "bounding_boxes_xyxy", "reported_overlap_area_percent")
    if any(not isinstance(stage.get(key), list) or len(stage[key]) != count for key in fields):
        raise ValueError("Wrong metric vector size in " + name)
    if any(type(flag) is not bool for flag in stage["official_valid"]):
        raise ValueError("Invalid validity flags in " + name)
    valid = [i for i, flag in enumerate(stage["official_valid"]) if flag]
    best = max(valid, key=lambda i: stage["utilization"][i]) if valid else None
    utilization = stage["utilization"][best] if best is not None else None
    if (stage.get("official_valid_count") != len(valid) or stage.get("best_official_valid_index") != best
            or stage.get("best_official_valid_utilization") != utilization):
        raise ValueError("Metric selection mismatch in " + name)
    for key in ("utilization", "reported_overlap_area_percent"):
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in stage[key]):
            raise ValueError("Invalid metric value in " + name)
    for box in stage["bounding_boxes_xyxy"]:
        if (not isinstance(box, list) or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)):
            raise ValueError("Invalid bounding box in " + name)


def selected_metrics(metrics, candidate_index):
    selected = {}
    for stage in ("raw", "enhanced", "gt_input", "gt_enhanced"):
        values = metrics[stage]
        index = candidate_index if stage in ("raw", "enhanced") else 0
        selected[stage] = {"candidate_index": index,
                           "official_valid": values["official_valid"][index],
                           "utilization": values["utilization"][index],
                           "bounding_box_xyxy": values["bounding_boxes_xyxy"][index],
                           "reported_overlap_area_percent": values["reported_overlap_area_percent"][index]}
    return selected


def validate_geometry(inputs, geometry, index):
    if inputs.get("sample_index") != index:
        raise ValueError("Saved input has a different sample index")
    ids = inputs.get("polygon_ids")
    polygons = geometry.get("official_max_contours")
    if (not isinstance(ids, list) or not ids or not isinstance(polygons, list)
            or len(polygons) != 440 or geometry.get("coordinate_scale_from_polygon_txt") != 10):
        raise ValueError("Unexpected input geometry")
    for polygon_id in ids:
        if require_int(polygon_id, "polygon ID") >= len(polygons):
            raise ValueError("Polygon ID is outside the saved contour table")
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError("Invalid saved polygon contour")
        for vertex in polygon:
            if (not isinstance(vertex, list) or len(vertex) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in vertex)):
                raise ValueError("Invalid saved polygon vertex")
    return ids, polygons


def load_actions(torch, path, shape, candidate_index=None):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
        raise ValueError("Unexpected saved tensor shape: " + path.name)
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
        raise ValueError("Nonfinite or non-floating actions: " + path.name)
    if candidate_index is not None:
        tensor = tensor[candidate_index]
    if (tensor[:, 2:].square().sum(dim=1) == 0).any().item():
        raise ValueError("Zero rotation direction: " + path.name)
    return tensor.tolist(), str(tensor.dtype)


def freeze_case(torch, results, index, group, baseline):
    case = results / ("sample_%03d" % index)
    metadata_path = case / "metadata.json"
    metadata = read_json(metadata_path)
    for key, expected in (("status", "completed"), ("sample_index", index), ("seed", 3407),
                          ("candidates", 512), ("steps", 128),
                          ("model_mode", "upstream_default_train_dropout")):
        if metadata.get(key) != expected:
            raise ValueError("Unexpected saved case configuration: " + key)
    recorded_hashes = metadata.get("output_sha256", {})
    sources = []
    for name in CASE_FILES:
        expected = recorded_hashes.get(name)
        require_hash(expected, name)
        path = case / name
        if case.resolve() not in path.resolve().parents or digest(path) != expected:
            raise ValueError("Saved output hash mismatch: sample_%03d/%s" % (index, name))
        sources.append({"path": "sample_%03d/%s" % (index, name), "sha256": expected})
    sources.append({"path": "sample_%03d/metadata.json" % index, "sha256": digest(metadata_path)})
    inputs, geometry, metrics = (read_json(case / name) for name in
                                 ("input_sample.json", "input_geometry.json", "metrics.json"))
    ids, polygons = validate_geometry(inputs, geometry, index)
    for stage in ("raw", "enhanced", "gt_input", "gt_enhanced"):
        validate_metrics(metrics[stage], 512 if stage in ("raw", "enhanced") else 1, stage)
        if stage in ("raw", "enhanced"):
            for key, expected in baseline["rows"][index][stage].items():
                if metrics[stage].get(key) != expected:
                    raise ValueError("Saved metrics disagree with baseline summary")
    chosen, policy = selection(metrics)
    require_int(chosen, "selected candidate")
    audit = baseline["audits"][index]
    if audit.get("selected_candidate") != chosen or audit.get("selection_policy") != policy:
        raise ValueError("Frozen candidate selection disagrees with diagnosis")
    if (audit.get("official_selected_utilization") != metrics["enhanced"]["utilization"][chosen]
            or audit.get("official_selected_thresholded_overlap_percent")
            != metrics["enhanced"]["reported_overlap_area_percent"][chosen]):
        raise ValueError("Selected candidate metrics disagree with diagnosis")
    actions, dtypes = {}, {}
    for field, filename, candidate in (("raw_actions", "predictions_raw.pt", chosen),
                                       ("enhanced_actions", "predictions_enhanced.pt", chosen),
                                       ("gt_actions", "gt_input.pt", None),
                                       ("gt_enhanced_actions", "gt_enhanced.pt", None)):
        shape = (512, len(ids), 4) if candidate is not None else (len(ids), 4)
        actions[field], dtypes[field] = load_actions(torch, case / filename, shape, candidate)
    if inputs.get("gt_actions_official_scaled") != actions["gt_actions"]:
        raise ValueError("GT tensor disagrees with saved official-scaled input")
    # Recheck small saved files and tensors after reads, catching concurrent edits.
    for source in sources:
        if digest(results / source["path"]) != source["sha256"]:
            raise ValueError("Saved case changed during freezing")
    return {"schema_version": 1, "sample_index": index, "group": group,
            "candidate_index": chosen, "selection_policy": policy,
            "polygons": polygons, "polygon_ids": ids, **actions,
            "action_columns": inputs.get("action_columns"), "tensor_dtypes": dtypes,
            "input_configuration": {"coordinate_scale_from_polygon_txt": 10,
                                    "official_direction_scale": inputs.get("official_direction_scale"),
                                    "padding_mask": inputs.get("padding_mask")},
            "baseline_metrics": {**metrics, "selected": selected_metrics(metrics, chosen),
                                 "diagnosis": audit},
            "baseline_source_files": sources}


def freeze(results, diagnosis_path, output):
    results, diagnosis_path, output = (path.expanduser().resolve() for path in
                                        (results, diagnosis_path, output))
    if diagnosis_path.is_dir():
        diagnosis_path = diagnosis_path / "diagnosis.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite a previous freeze")
    if (results == output or results in output.parents or output in results.parents
            or output == diagnosis_path or output in diagnosis_path.parents):
        raise ValueError("Freeze output must be separate from baseline and diagnosis")
    baseline = validate_baseline(results, diagnosis_path)
    initial_hashes = {name: digest(path) for name, path in baseline["files"].items()}
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required only for CPU weights-only loading") from exc
    cases = []
    for index, group in ([(index, "failure_development") for index in FAILED_SAMPLES]
                         + [(index, "success_regression") for index in baseline["successes"]]):
        cases.append(freeze_case(torch, results, index, group, baseline))
    for name, path in baseline["files"].items():
        if digest(path) != initial_hashes[name]:
            raise ValueError("Baseline evidence changed during freezing: " + name)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    entries = []
    for case in cases:
        relative = "cases/sample_%03d.json" % case["sample_index"]
        write_json(output / relative, case)
        entries.append({"sample_index": case["sample_index"], "group": case["group"],
                        "candidate_index": case["candidate_index"], "path": relative,
                        "sha256": digest(output / relative)})
    manifest = {"schema_version": 1, "status": "frozen",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "16 fixed development/regression cases; not independent generalization evidence; no inference or training",
                "configuration": {**baseline["configuration"], "dental_strip_height": 1200,
                                  "reference_width": 8000, "width_policy": "no arbitrary widening",
                                  "reference_geometry": "float64; original diagnosis retained without recomputation",
                                  "strict_overlap_area_tolerance": 1e-9,
                                  "tensor_load": "torch.load(map_location='cpu', weights_only=True)"},
                "selection": {"failure_samples": FAILED_SAMPLES,
                              "success_samples": baseline["successes"],
                              "success_population_count": 120,
                              "success_population_order": "ascending sample_index among official-valid enhanced cases",
                              "success_positions_zero_based": baseline["positions"],
                              "success_position_algorithm": "p_i = (i * (N - 1)) // (K - 1), i=0..K-1, N=120, K=8",
                              "candidate_policy": "diagnose_results.selection; same enhanced-selected index for raw and enhanced",
                              "failure_selection_note": "Minimum official thresholded overlap then highest utilization; not minimum exact overlap"},
                "baseline_evidence": {
                    "files": [{"root": "diagnosis" if name == "diagnosis" else "results",
                               "path": name + ".json", "sha256": sha}
                              for name, sha in initial_hashes.items()],
                    "recorded_frozen_input_mapping_canonical_sha256": canonical_digest(baseline["plan"]["input_sha256"]),
                    "portable_frozen_input_inventory_sha256": canonical_digest(baseline["inventory"]),
                    "frozen_input_inventory": baseline["inventory"],
                    "verification": baseline["verification"],
                    "validation_scope": "Completed verified baseline, summary recomputation, diagnosis binding, selected saved-file hashes/configuration; original source files and unselected output tensors are not reverified here"},
                "implementation_files": [{"path": name, "sha256": digest(Path(__file__).with_name(name))}
                                         for name in ("freeze_collision_inputs.py", "diagnose_results.py", "run_benchmark.py")],
                "cases": entries}
    write_json(output / "manifest.json", manifest)
    manifest_hash = digest(output / "manifest.json")
    with (output / "manifest.sha256").open("x", encoding="ascii") as target:
        target.write(manifest_hash + "  manifest.json\n")
    return {"status": "frozen", "cases": len(entries),
            "failure_samples": FAILED_SAMPLES, "success_samples": baseline["successes"],
            "manifest_sha256": manifest_hash}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Completed verified upstream_results_128 directory")
    parser.add_argument("--diagnosis", type=Path, required=True,
                        help="Matching diagnosis.json file or its directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="New, nonexistent freeze directory")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(freeze(args.results, args.diagnosis, args.output), indent=2))
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
