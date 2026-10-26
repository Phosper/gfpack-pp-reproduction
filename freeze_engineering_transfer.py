#!/usr/bin/env python3
"""Freeze a verified upstream baseline as 128 fixed engineering inputs.

The 16 development JSON files are copied byte for byte. Every case is freshly
checked against the saved baseline by freeze_collision_inputs.freeze_case;
there is no inference, resampling, training, or change to candidate selection.
PyTorch is loaded lazily and only used for CPU weights-only tensor reads.
The validation helpers have no PyTorch dependency and work on a portable copy.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

from check_collision_saved_cases import (
    EXPECTED_MANIFEST_SHA256, FAILURE_SAMPLES, SUCCESS_SAMPLES,
    checked_case, validate_frozen,
)
from diagnose_results import selection
from freeze_collision_inputs import (
    TOTAL_SAMPLES, canonical_digest, freeze_case, read_json, require_hash,
    require_int, validate_baseline, validate_geometry, validate_metrics,
    write_json,
)
from run_benchmark import digest


CONTRACT_TYPE = "engineering_transfer_128x1"
DEVELOPMENT_SAMPLES = sorted(FAILURE_SAMPLES + SUCCESS_SAMPLES)
TRANSFER_SAMPLES = sorted(set(range(TOTAL_SAMPLES)) - set(DEVELOPMENT_SAMPLES))
CONTRACT_FILE = "evaluation_contract.json"
DEVELOPMENT_MANIFEST = "source_manifests/development_manifest.json"
ACTION_FIELDS = ("raw_actions", "enhanced_actions", "gt_actions", "gt_enhanced_actions")
IMPLEMENTATION_FILES = (
    "freeze_engineering_transfer.py", "freeze_collision_inputs.py",
    "check_collision_saved_cases.py", "diagnose_results.py", "run_benchmark.py",
    "compact_portfolio.py", "compact_layout.py", "translation_intervals.py",
    "postprocess_metrics.py", "collision_repair.hpp", "build_collision_probe.py",
)


def evaluation_contract():
    """Return the fixed, pre-results evaluation policy as portable JSON data."""
    return {
        "schema_version": 1,
        "contract_type": CONTRACT_TYPE,
        "scope": "Engineering transfer of one fixed saved candidate per validation sample; the 16 development samples remain development evidence; the other 112 are transfer checks, not independent training or resampling evidence",
        "sample_indices": list(range(TOTAL_SAMPLES)),
        "development_samples": DEVELOPMENT_SAMPLES,
        "transfer_samples": TRANSFER_SAMPLES,
        "planned_prediction_inputs": 128,
        "planned_gt_inputs": 128,
        "planned_total_inputs": 256,
        "candidates_per_sample": 1,
        "new_inference": False,
        "training": False,
        "resampling": False,
        "candidate_selection": {
            "function": "diagnose_results.selection",
            "policy": "Best official-valid enhanced utilization; when absent, minimum official thresholded overlap then highest utilization; raw and enhanced always use that same stored candidate index",
            "failure_note": "The fallback is not necessarily the minimum exact-overlap candidate",
            "prediction_input": "raw_actions of the frozen selected candidate",
            "gt_input": "original gt_actions, evaluated separately as a paired reference",
        },
        "geometry": {
            "height": 1200.0,
            "horizontal_walls": None,
            "reference_width": 8000.0,
            "strict_area_tolerance": 1e-9,
            "boundary_tolerance": 1e-9,
            "official_reference_area_threshold": 50.0,
            "stored_output_precision": "float32",
            "independent_verification_precision": "float64",
            "strict_success": "Independent verification of stored output: all pairwise intersection areas <= 1e-9 and all pieces inside the fixed height strip",
            "reference_threshold_note": "50 is retained for reference reporting and does not replace strict feasibility",
        },
        "solver": {
            "seconds_per_attempt": 45.0,
            "sweeps_per_attempt": 4,
            "starts": ["raw", "legacy_repair"],
            "uniform_policy": "Apply the same two-start rule to all 128 predictions and all 128 original GTs; no case-specific exception",
            "second_start": "Deterministic legacy overlap repair of the same frozen input; prediction seed never comes from GT or another candidate",
            "legacy_seed_area_eps": 1.0,
            "legacy_seed_max_iterations": 2000,
            "selection": "Choose minimum horizontal span among strictly feasible stored float32 endpoints; raw wins exact ties",
            "identical_starts": "Explicitly reuse the raw attempt when the two stored float32 starts are equal",
            "algorithm_change": False,
        },
        "failure_policy": {
            "retain_all_attempts": True,
            "retain_timeouts": True,
            "retain_partial_outputs": True,
            "illegal_layout_can_win": False,
            "timeout_can_be_silently_dropped": False,
            "strict_failure_utilization": 0.0,
            "all_case_mean_denominator": {"pred": 128, "gt": 128},
            "group_denominators": {"development": 16, "transfer": 112},
            "paired_comparison": "Only paired valid outputs may enter valid-only comparisons; full-scope metrics retain all failures as zero",
        },
        "timing": {
            "clock": "wall-clock seconds",
            "required_separate_components": [
                "seed_generation_seconds", "solver_seconds", "io_seconds",
                "verification_seconds", "total_wall_seconds",
            ],
            "solver_seconds": "Sum of distinct attempts, without double-counting identical-start reuse",
            "end_to_end": "Include seed generation, solver calls, I/O and independent verification; preserve per-stage and total timing for failures and timeouts",
            "legacy_seed_build": "Runner binds the actual isolated extension build manifest, extension binary and source manifests before evaluation",
        },
        "reporting": {
            "groups": ["all_128", "development_16", "transfer_112"],
            "sources": ["pred", "gt"],
            "required": ["strict_success_count", "failure_count", "timeout_count",
                         "all_input_mean_utilization_failure_as_zero",
                         "valid_only_mean_utilization", "timing_components"],
        },
    }


def _safe_file(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Expected a portable relative evidence path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or ".." in parts.parts or ":" in relative:
        raise ValueError("Unsafe evidence path")
    path = (root / Path(*parts.parts)).resolve()
    if root not in path.parents:
        raise ValueError("Evidence path escapes its frozen root")
    return path


def _group(index):
    if index in FAILURE_SAMPLES:
        return "failure_development"
    if index in SUCCESS_SAMPLES:
        return "success_regression"
    return "transfer"


def _check_actions(case, field):
    values = case.get(field)
    if (not isinstance(values, list) or len(values) != len(case["polygon_ids"])
            or any(not isinstance(row, list) or len(row) != 4
                   or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
                   or (row[2] == 0 and row[3] == 0) for row in values)):
        raise ValueError("Invalid frozen actions: " + field)


def checked_transfer_case(root, entry):
    """Check one manifest entry and its JSON without importing torch."""
    root = Path(root).expanduser().resolve()
    if not isinstance(entry, dict):
        raise ValueError("Expected a frozen case manifest entry")
    index = require_int(entry.get("sample_index"), "sample_index")
    candidate = require_int(entry.get("candidate_index"), "candidate_index")
    if index >= TOTAL_SAMPLES or candidate >= 512:
        raise ValueError("Case sample or candidate index outside the baseline")
    evaluation_group = "development" if index in DEVELOPMENT_SAMPLES else "transfer"
    if (entry.get("path") != "cases/sample_%03d.json" % index
            or entry.get("group") != _group(index)
            or entry.get("evaluation_group") != evaluation_group):
        raise ValueError("Unexpected frozen case path or group")
    require_hash(entry.get("sha256"), "frozen case")
    path = _safe_file(root, entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError("Frozen transfer case SHA-256 mismatch")
    case = read_json(path)
    if case.get("schema_version") != 1:
        raise ValueError("Unknown frozen case schema")
    for key in ("sample_index", "candidate_index", "group", "selection_policy"):
        if case.get(key) != entry.get(key):
            raise ValueError("Frozen case identity disagrees with manifest: " + key)
    configuration = case.get("input_configuration", {})
    validate_geometry({"sample_index": index, "polygon_ids": case.get("polygon_ids")},
                      {"official_max_contours": case.get("polygons"),
                       "coordinate_scale_from_polygon_txt": configuration.get("coordinate_scale_from_polygon_txt")},
                      index)
    for field in ACTION_FIELDS:
        _check_actions(case, field)
    metrics = case.get("baseline_metrics", {})
    for stage in ("raw", "enhanced", "gt_input", "gt_enhanced"):
        validate_metrics(metrics.get(stage, {}), 512 if stage in ("raw", "enhanced") else 1, stage)
    chosen, policy = selection(metrics)
    audit = metrics.get("diagnosis", {})
    if (candidate != chosen or case.get("selection_policy") != policy
            or audit.get("sample_index") != index
            or audit.get("selected_candidate") != chosen or audit.get("selection_policy") != policy):
        raise ValueError("Frozen candidate disagrees with original diagnosis selection")
    sources = case.get("baseline_source_files")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Missing saved baseline source provenance")
    source_paths = set()
    for source in sources:
        require_hash(source.get("sha256"), "baseline case source")
        relative = source.get("path")
        if (not isinstance(relative, str) or relative in source_paths
                or not relative.startswith("sample_%03d/" % index)):
            raise ValueError("Invalid baseline case source provenance")
        _safe_file(root, relative)
        source_paths.add(relative)
    return path, case


def validate_transfer_frozen(root):
    """Validate a portable 128-case freeze; original remote paths are unnecessary."""
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest_sha = digest(manifest_path)
    sidecar = (root / "manifest.sha256").read_text(encoding="ascii").split()
    if sidecar != [manifest_sha, "manifest.json"]:
        raise ValueError("Frozen transfer manifest sidecar mismatch")
    manifest = read_json(manifest_path)
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "frozen"
            or manifest.get("experiment_type") != CONTRACT_TYPE
            or manifest.get("sample_indices") != list(range(TOTAL_SAMPLES))
            or manifest.get("development_samples") != DEVELOPMENT_SAMPLES
            or manifest.get("transfer_samples") != TRANSFER_SAMPLES):
        raise ValueError("Unexpected engineering transfer freeze schema or coverage")
    contract_entry = manifest.get("evaluation_contract", {})
    if contract_entry.get("path") != CONTRACT_FILE:
        raise ValueError("Missing frozen evaluation contract")
    require_hash(contract_entry.get("sha256"), "evaluation contract")
    contract_path = _safe_file(root, CONTRACT_FILE)
    if (digest(contract_path) != contract_entry["sha256"]
            or read_json(contract_path) != evaluation_contract()):
        raise ValueError("Evaluation contract changed from the fixed 128x1 policy")
    development = manifest.get("development_evidence", {})
    if (development.get("path") != DEVELOPMENT_MANIFEST
            or development.get("sha256") != EXPECTED_MANIFEST_SHA256):
        raise ValueError("Development source manifest is not the verified original")
    development_path = _safe_file(root, DEVELOPMENT_MANIFEST)
    if digest(development_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Development source manifest bytes changed")
    original = read_json(development_path)
    original_entries = {e["sample_index"]: e for e in original["cases"]}
    entries = manifest.get("cases")
    if (not isinstance(entries, list) or len(entries) != TOTAL_SAMPLES
            or [entry.get("sample_index") for entry in entries] != list(range(TOTAL_SAMPLES))):
        raise ValueError("Missing, duplicated or unordered 128-case coverage")
    evidence = manifest.get("baseline_evidence", {})
    old_evidence = original.get("baseline_evidence", {})
    for key in ("recorded_frozen_input_mapping_canonical_sha256",
                "portable_frozen_input_inventory_sha256", "frozen_input_inventory"):
        if evidence.get(key) != old_evidence.get(key):
            raise ValueError("Baseline inputs differ from the original development freeze")
    inventory = evidence.get("frozen_input_inventory")
    if (not isinstance(inventory, list) or not inventory
            or canonical_digest(inventory) != evidence["portable_frozen_input_inventory_sha256"]):
        raise ValueError("Invalid portable frozen source inventory")
    files = evidence.get("files")
    if files != old_evidence.get("files"):
        raise ValueError("Baseline source manifests differ from development evidence")
    source_checks = evidence.get("source_checks")
    if not isinstance(source_checks, list) or len(source_checks) != len(inventory):
        raise ValueError("Missing baseline resource validation evidence")
    by_source = {row.get("source_key_sha256"): row for row in inventory}
    checked_sources = set()
    for row in source_checks:
        key = row.get("source_key_sha256")
        if key not in by_source or key in checked_sources:
            raise ValueError("Invalid or duplicated baseline resource evidence")
        expected = by_source[key]
        if row.get("path") != expected["path"] or row.get("sha256") != expected["sha256"]:
            raise ValueError("Baseline resource evidence hash mismatch")
        if row.get("validation") == "original_resource_hash_verified":
            if row.get("observed_sha256") != expected["sha256"]:
                raise ValueError("A baseline resource was not verified")
        elif row.get("validation") == "historical_runner_bound_to_verified_baseline":
            if row.get("path") not in ("runner/run_inference.py", "runner/run_benchmark.py"):
                raise ValueError("Historical-source exception is only for old inference runners")
        else:
            raise ValueError("Unknown baseline source validation status")
        checked_sources.add(key)
    for entry in entries:
        checked_transfer_case(root, entry)
        index = entry["sample_index"]
        if index in original_entries:
            for key in ("path", "sha256", "group", "candidate_index"):
                if entry.get(key) != original_entries[index].get(key):
                    raise ValueError("Development case bytes or identity changed")
    return manifest, manifest_sha


def _source_checks(baseline):
    """Verify original resources while recording historical runner custody.

    Later experiments may have replaced the original inference entry scripts.
    Those two scripts are never executed here: their original hashes remain
    bound by the completed verification and unchanged development manifest.
    Data, weights, upstream sources and native baseline extensions must still
    match their original frozen hashes, without exceptions.
    """
    inventory = {row["source_key_sha256"]: row for row in baseline["inventory"]}
    checks, snapshots = [], []
    for source, expected in sorted(baseline["plan"]["input_sha256"].items()):
        key = hashlib.sha256(source.encode("utf-8")).hexdigest()
        row = dict(inventory[key])
        path = Path(source).expanduser()
        observed = digest(path) if path.is_file() else None
        historical_runner = row["path"] in ("runner/run_inference.py", "runner/run_benchmark.py")
        if observed != expected and not historical_runner:
            raise ValueError("Original baseline resource hash mismatch: " + row["path"])
        row.update(observed_sha256=observed,
                   validation=("original_resource_hash_verified" if observed == expected
                               else "historical_runner_bound_to_verified_baseline"))
        checks.append(row)
        snapshots.append((path, observed))
    return checks, snapshots


def _snapshot_check(snapshots):
    for path, expected in snapshots:
        observed = digest(path) if path.is_file() else None
        if observed != expected:
            raise ValueError("A frozen source changed during input preparation: " + path.name)


def freeze(results, diagnosis_path, development_frozen, output):
    results, diagnosis_path, development_frozen, output = (
        Path(path).expanduser().resolve()
        for path in (results, diagnosis_path, development_frozen, output))
    if diagnosis_path.is_dir():
        diagnosis_path /= "diagnosis.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing engineering freeze")
    for root in (results, development_frozen):
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("Freeze output must be separate from original evidence")
    if output == diagnosis_path or output in diagnosis_path.parents:
        raise ValueError("Freeze output must be separate from the diagnosis")
    baseline = validate_baseline(results, diagnosis_path)
    development_manifest, development_sha = validate_frozen(development_frozen)
    old_evidence = development_manifest["baseline_evidence"]
    baseline_hashes = {name: digest(path) for name, path in baseline["files"].items()}
    evidence_files = [
        {"root": "diagnosis" if name == "diagnosis" else "results",
         "path": name + ".json", "sha256": sha}
        for name, sha in baseline_hashes.items()
    ]
    if (evidence_files != old_evidence["files"]
            or canonical_digest(baseline["plan"]["input_sha256"])
            != old_evidence["recorded_frozen_input_mapping_canonical_sha256"]
            or baseline["inventory"] != old_evidence["frozen_input_inventory"]):
        raise ValueError("Baseline no longer matches the verified development evidence")
    source_checks, snapshots = _source_checks(baseline)
    snapshots.extend((path, baseline_hashes[name]) for name, path in baseline["files"].items())
    snapshots.append((development_frozen / "manifest.json", development_sha))
    implementation = []
    for name in IMPLEMENTATION_FILES:
        path = Path(__file__).resolve().with_name(name)
        sha = digest(path)
        implementation.append({"path": name, "sha256": sha})
        snapshots.append((path, sha))
    old_entries = {entry["sample_index"]: entry for entry in development_manifest["cases"]}
    if sorted(old_entries) != DEVELOPMENT_SAMPLES:
        raise ValueError("The development group is not the fixed original 16 cases")

    # Write the contract before extracting any new case. A failed preparation
    # deliberately leaves incomplete evidence; its output path cannot be reused.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / CONTRACT_FILE, evaluation_contract())
    contract_sha = digest(output / CONTRACT_FILE)
    (output / "cases").mkdir()
    (output / "source_manifests").mkdir()
    development_bytes = (development_frozen / "manifest.json").read_bytes()
    if hashlib.sha256(development_bytes).hexdigest() != development_sha:
        raise ValueError("Development manifest changed while being copied")
    with (output / DEVELOPMENT_MANIFEST).open("xb") as target:
        target.write(development_bytes)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required only for CPU weights-only tensor loading") from exc

    entries = []
    for index in range(TOTAL_SAMPLES):
        case = freeze_case(torch, results, index, _group(index), baseline)
        relative = "cases/sample_%03d.json" % index
        if index in old_entries:
            source_path, original = checked_case(development_frozen, old_entries[index])
            # Compare every field, including candidate, raw/GT arrays, contours,
            # full per-candidate metrics and saved-file provenance. The old
            # bytes themselves are retained, even if JSON formatting differs.
            if original != case:
                raise ValueError("Development case differs from freshly validated baseline: %03d" % index)
            content = source_path.read_bytes()
            if hashlib.sha256(content).hexdigest() != old_entries[index]["sha256"]:
                raise ValueError("Development case changed during its byte copy")
            with (output / relative).open("xb") as target:
                target.write(content)
            snapshots.append((source_path, old_entries[index]["sha256"]))
        else:
            write_json(output / relative, case)
        entry = {
            "sample_index": index, "group": case["group"],
            "evaluation_group": "development" if index in old_entries else "transfer",
            "candidate_index": case["candidate_index"],
            "selection_policy": case["selection_policy"],
            "path": relative, "sha256": digest(output / relative),
        }
        checked_transfer_case(output, entry)
        entries.append(entry)
        print(json.dumps({"sample_index": index, "group": entry["evaluation_group"],
                          "candidate_index": entry["candidate_index"],
                          "development_bytes_preserved": index in old_entries}), flush=True)
    _snapshot_check(snapshots)
    if digest(output / CONTRACT_FILE) != contract_sha:
        raise ValueError("Predeclared evaluation contract changed during freezing")
    manifest = {
        "schema_version": 1, "status": "frozen", "experiment_type": CONTRACT_TYPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": evaluation_contract()["scope"],
        "sample_indices": list(range(TOTAL_SAMPLES)),
        "development_samples": DEVELOPMENT_SAMPLES,
        "transfer_samples": TRANSFER_SAMPLES,
        "configuration": {**baseline["configuration"], "candidates_per_sample": 1,
                          "dental_strip_height": 1200, "reference_width": 8000,
                          "strict_overlap_area_tolerance": 1e-9,
                          "official_overlap_area_threshold": 50,
                          "tensor_load": "torch.load(map_location='cpu', weights_only=True)"},
        "evaluation_contract": {"path": CONTRACT_FILE, "sha256": contract_sha},
        "development_evidence": {"path": DEVELOPMENT_MANIFEST, "sha256": development_sha,
                                 "case_bytes_preserved": True, "cases": 16},
        "baseline_evidence": {
            "files": evidence_files,
            "recorded_frozen_input_mapping_canonical_sha256": canonical_digest(baseline["plan"]["input_sha256"]),
            "portable_frozen_input_inventory_sha256": canonical_digest(baseline["inventory"]),
            "frozen_input_inventory": baseline["inventory"],
            "verification": baseline["verification"],
            "source_checks": source_checks,
            "validation_scope": "Completed and verified baseline; summary recomputation and diagnosis binding; every case's saved input, raw/enhanced prediction, input/enhanced GT tensor and metadata checked; all original data, weights, upstream sources and native baseline resources rehashed; historical inference-runner replacements explicitly recorded and never executed",
            "seed_build_provenance": "Bound separately by the evaluation runner to its actual isolated extension build and copied source manifest; no seed build is loaded during this freeze",
        },
        "implementation_files": implementation,
        "cases": entries,
    }
    write_json(output / "manifest.json", manifest)
    manifest_sha = digest(output / "manifest.json")
    with (output / "manifest.sha256").open("x", encoding="ascii") as target:
        target.write(manifest_sha + "  manifest.json\n")
    validate_transfer_frozen(output)
    return {"status": "frozen", "experiment_type": CONTRACT_TYPE,
            "cases": 128, "development_cases": 16, "transfer_cases": 112,
            "manifest_sha256": manifest_sha, "evaluation_contract_sha256": contract_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Completed verified upstream-default 128-case baseline")
    parser.add_argument("--diagnosis", type=Path, required=True,
                        help="Matching diagnosis.json or its directory")
    parser.add_argument("--development-frozen", type=Path, required=True,
                        help="Previously verified frozen 16-case input directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="New, nonexistent portable engineering freeze directory")
    args = parser.parse_args(argv)
    try:
        result = freeze(args.results, args.diagnosis, args.development_frozen, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
