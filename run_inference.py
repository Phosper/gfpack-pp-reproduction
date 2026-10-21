#!/usr/bin/env python3
"""Run one official GFPack++ validation sample on one CUDA GPU, without DDP.

Only load the checkpoint/PKL obtained from the trusted official release: pickle
deserialization is not a sandbox. Defaults use 512 candidates and 128 sampler
steps. Smaller runs must be explicitly marked --smoke and are not a benchmark.
The upstream checkout is read-only; every invocation needs a new output path.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time
import traceback


SEED = 3407
POLYGON_COUNT = 440
OFFICIAL_EPS = 50.0
NOMINAL_WIDTH = 8000.0
NOMINAL_HEIGHT = 1205.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def command_snapshot(command, cwd=None):
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=10,
            check=False,
        )
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path,
                        help="Official GFPack-pp checkout; not modified")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Trusted official score_model.pth state_dict")
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Trusted validation PKL, containing IDs then actions")
    parser.add_argument("--polygons", required=True, type=Path,
                        help="Official polygon directory with 0.txt through 439.txt")
    parser.add_argument("--extensions", type=Path,
                        help="Optional directory of locally built calutil/rmspacing extensions")
    parser.add_argument("--output", required=True, type=Path,
                        help="New, nonexistent result directory")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--smoke", action="store_true",
                        help="Mark a reduced run as smoke only, not benchmark evidence")
    args = parser.parse_args(argv)
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    if args.candidates < 1:
        parser.error("--candidates must be at least 1")
    if args.steps < 2:
        parser.error("--steps must be at least 2 (required by the official sampler)")
    if (args.candidates < 512 or args.steps < 128) and not args.smoke:
        parser.error("Reduced candidate/step runs require --smoke")
    for name in ("repo", "checkpoint", "dataset", "polygons", "output", "extensions"):
        if getattr(args, name) is None:
            continue
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def check_paths(args):
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite existing output: %s" % args.output)
    required = [args.checkpoint, args.dataset]
    required.extend(args.repo / "src" / name for name in
                    ("__init__.py", "evaluate.py", "sde.py", "score_model.py",
                     "gnn_feature.py", "geometry_utils.py"))
    required.extend(args.polygons / ("%d.txt" % i)
                    for i in range(POLYGON_COUNT))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files (%d):\n%s" %
                                (len(missing), "\n".join(missing[:20])))
    if args.checkpoint.stat().st_size == 0 or args.dataset.stat().st_size == 0:
        raise ValueError("Checkpoint and dataset must be nonempty")
    # Results must not change the pinned upstream tree or contain a source path.
    protected = [args.repo, args.checkpoint, args.dataset, args.polygons, Path(__file__).resolve()]
    for path in protected:
        if args.output == path or args.output in path.parents:
            raise ValueError("Output would contain a protected source: %s" % path)
    if args.repo in args.output.parents:
        raise ValueError("Choose an output directory outside the upstream checkout")


def checked_sample(torch, evaluation, dataset, sample_index):
    with dataset.open("rb") as source:
        all_ids = pickle.load(source)
        all_actions = pickle.load(source)
    if len(all_ids) != len(all_actions):
        raise ValueError("PKL sample counts disagree between IDs and actions")
    if sample_index >= len(all_ids):
        raise IndexError("sample-index %d is outside dataset [0, %d)" %
                         (sample_index, len(all_ids)))
    ids_untyped = torch.as_tensor(all_ids[sample_index]).detach().cpu()
    original_gt = torch.as_tensor(all_actions[sample_index],
                                  dtype=torch.float32).detach().cpu()
    if ids_untyped.ndim == 2 and ids_untyped.shape[0] == 1:
        ids_untyped = ids_untyped.squeeze(0)
    if original_gt.ndim == 3 and original_gt.shape[0] == 1:
        original_gt = original_gt.squeeze(0)
    if ids_untyped.ndim != 1 or ids_untyped.numel() < 1:
        raise ValueError("Expected a nonempty, one-dimensional polygon ID sequence")
    if not torch.isfinite(ids_untyped).all().item():
        raise ValueError("Polygon IDs contain nonfinite values")
    ids = ids_untyped.to(torch.int64)
    if not torch.equal(ids_untyped, ids.to(ids_untyped.dtype)):
        raise ValueError("Polygon IDs must be integers")
    if ids.min().item() < 0 or ids.max().item() >= POLYGON_COUNT:
        raise ValueError("Polygon IDs must be within [0, 440)")
    if tuple(original_gt.shape) != (ids.numel(), 4):
        raise ValueError("GT actions must have shape (polygon_count, 4)")
    if not torch.isfinite(original_gt).all().item():
        raise ValueError("GT actions contain nonfinite values")
    if (original_gt[:, 2:].square().sum(dim=1) == 0).any().item():
        raise ValueError("GT contains a zero rotation direction")
    ids_list, scaled_actions = evaluation.padData(
        [ids.tolist()], [copy.deepcopy(original_gt.tolist())])
    mask = torch.tensor(evaluation.genPaddingMask(ids_list)[0], dtype=torch.float32)
    gt = torch.tensor(scaled_actions[0], dtype=torch.float32)
    return ids, original_gt.clone(), gt, mask, len(all_ids)


def ensure_finite(torch, value, name):
    if not torch.isfinite(value).all().item():
        raise ValueError("%s contains nonfinite values; partial outputs retained" % name)


def native_metrics(torch, evaluation, polygons, ids, actions):
    id_batches = [ids.tolist() for _ in range(len(actions))]
    util, valid, bbox, overlap = evaluation.cal_util(
        polygons, id_batches, actions.tolist(), OFFICIAL_EPS)
    for value, name in ((util, "utilization"), (bbox, "bounding boxes"),
                        (overlap, "overlap percentage")):
        ensure_finite(torch, value, "Official " + name)
    indices = [index for index, flag in enumerate(valid.tolist()) if flag]
    best_index = max(indices, key=lambda index: util[index].item()) if indices else None
    return {"candidate_count": len(actions), "official_valid_count": len(indices),
            "best_official_valid_index": best_index,
            "best_official_valid_utilization":
                float(util[best_index]) if best_index is not None else None,
            "official_valid": valid.tolist(), "utilization": util.tolist(),
            "bounding_boxes_xyxy": bbox.tolist(),
            "reported_overlap_area_percent": overlap.tolist()}


def exact_preview_audit(polygons, ids, actions):
    """Independent, zero-threshold geometric check of the displayed placement."""
    from shapely.geometry import Polygon

    placed = []
    for poly_id, action in zip(ids.tolist(), actions.tolist()):
        theta = math.atan2(action[3], action[2])
        cosine, sine = math.cos(theta), math.sin(theta)
        vertices = [(x * cosine - y * sine + action[0],
                     x * sine + y * cosine + action[1])
                    for x, y in polygons[poly_id]]
        placed.append(Polygon(vertices))
    if any(not poly.is_valid or poly.area <= 0 for poly in placed):
        return {"status": "invalid_polygon_geometry", "zero_overlap_pass": False}
    areas = [placed[i].intersection(placed[j]).area for i in range(len(placed))
             for j in range(i + 1, len(placed))]
    max_area, sum_area = max(areas, default=0.0), sum(areas)
    bounds = [poly.bounds for poly in placed]
    within_nominal_bounds = all(
        x0 >= -1e-9 and y0 >= -1e-9 and x1 <= NOMINAL_WIDTH + 1e-9
        and y1 <= NOMINAL_HEIGHT + 1e-9 for x0, y0, x1, y1 in bounds)
    return {"status": "measured", "max_pairwise_intersection_area": max_area,
            "sum_pairwise_intersection_area": sum_area,
            "zero_overlap_tolerance_area": 1e-9,
            "zero_overlap_pass": max_area <= 1e-9,
            "within_nominal_8000x1205_bounds": within_nominal_bounds,
            "nominal_bounds_note": "Reference from official vali_res arguments; "
            "current rmspacing implementation does not enforce these bounds."}


class PNGWriter:
    """Adapt the official visualize helper to a file instead of TensorBoard."""

    def __init__(self, path):
        self.path = path

    def add_figure(self, _tag, figure):
        for axis in figure.axes:
            axis.set_aspect("equal", adjustable="box")
        figure.tight_layout()
        figure.savefig(self.path, dpi=160)


def run(args):
    check_paths(args)
    # Delay all CUDA/scientific imports so --help and path checks work on a host
    # which has not yet installed the GPU environment.
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the GFPack GPU environment (PyTorch missing)") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA GPU is required; CPU fallback is disabled")
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    sys.path.insert(0, str(args.repo))
    sys.dont_write_bytecode = True
    import src
    if args.extensions:
        src.__path__.insert(0, str(args.extensions))
    from src import evaluate as evaluation
    from src.sde import init_sde, pc_sampler_state
    from src.score_model import PolygonPackingTransformer
    if Path(evaluation.__file__).resolve() != args.repo / "src" / "evaluate.py":
        raise RuntimeError("Imported src.evaluate is not from --repo")
    evaluation.POLYGON_DIR = args.polygons

    ids, original_gt, gt, mask, dataset_count = checked_sample(
        torch, evaluation, args.dataset, args.sample_index)
    polygons = evaluation.readAllPolys(POLYGON_COUNT)
    from shapely.geometry import Polygon
    for index, vertices in enumerate(polygons):
        array = np.asarray(vertices, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] != 2:
            raise ValueError("Polygon %d has invalid contour dimensions" % index)
        if not np.isfinite(array).all() or Polygon(vertices).area <= 0:
            raise ValueError("Polygon %d has nonfinite coordinates or zero area" % index)

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = {
        "status": "running", "started_utc": utc_now(),
        "run_kind": "SMOKE_ONLY" if args.smoke else "SINGLE_SAMPLE_INFERENCE",
        "not_a_full_benchmark": True, "seed": SEED,
        "sample_index": args.sample_index, "dataset_sample_count": dataset_count,
        "candidates": args.candidates, "steps": args.steps,
        "polygon_count": ids.numel(), "sde": "ve",
        "repo": str(args.repo), "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset), "output": str(args.output),
        "polygons": str(args.polygons),
        "extensions": str(args.extensions) if args.extensions else None,
        "command": sys.argv, "gpu_logical_device": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "execution": "Direct single GPU; model.eval(); torch.no_grad(); no DDP; no training",
        "differences_from_official_entrypoint": [
            "Fixed sample index instead of random.randint(0, 128)",
            "Explicit eval() disables dropout left active by the official entrypoint",
            "Raw tensors saved before vali_res can modify a cloned prediction",
            "PNG files use the official visualize helper with equal-aspect axes",
        ],
        "metric_semantics": {
            "official_overlap_eps_area": OFFICIAL_EPS,
            "official_valid_is_not_zero_overlap_or_boundary_verification": True,
            "official_utilization": "sum polygon areas / axis-aligned placement bounding-box area",
            "official_reported_overlap": "Only pairwise intersections greater than eps=50 are counted",
            "enhanced_gt": "vali_res appends GT to candidates and also applies gap removal to it",
            "geometry": "Official getMaxContour only; coordinates multiplied by 10 on loading",
        },
        "software": {"python": sys.version, "platform": platform.platform(),
                     "pytorch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version()},
        "hardware": {"machine": platform.machine(), "processor": platform.processor(),
                     "cpu_count": os.cpu_count()},
    }
    try:
        source_paths = [args.repo / "src" / name for name in
                        ("evaluate.py", "sde.py", "score_model.py", "gnn_feature.py",
                         "geometry_utils.py")]
        source_paths.extend(args.polygons / ("%d.txt" % i)
                            for i in range(POLYGON_COUNT))
        for module in (evaluation.calutil, evaluation.rmspacing):
            source_paths.append(Path(module.__file__).resolve())
        source_paths.extend([args.checkpoint, args.dataset, Path(__file__).resolve()])
        metadata["source_sha256"] = {str(path): sha256(path) for path in source_paths}
        metadata["git_head"] = command_snapshot(["git", "rev-parse", "HEAD"], args.repo)
        metadata["git_status"] = command_snapshot(["git", "status", "--porcelain"], args.repo)
        for package in ("torch-geometric", "numpy", "matplotlib", "shapely", "tensorboardX"):
            try:
                metadata["software"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                metadata["software"][package] = "unknown"
        metadata["nvidia_smi_before"] = command_snapshot([
            "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"])
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        props = torch.cuda.get_device_properties(device)
        metadata["hardware"].update({"gpu_name": props.name,
                                     "gpu_total_memory_bytes": props.total_memory,
                                     "gpu_compute_capability": [props.major, props.minor],
                                     "visible_gpu_count": torch.cuda.device_count()})
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_num_threads(16)
        evaluation.setRandomSeed(SEED)
        np.random.seed(SEED)
        torch.backends.cudnn.benchmark = False
        metadata["determinism_note"] = "RNG seeds fixed; not a cross-version bitwise determinism guarantee"

        write_json(args.output / "input_sample.json", {
            "sample_index": args.sample_index, "polygon_ids": ids.tolist(),
            "gt_actions_original_pkl": original_gt.tolist(),
            "gt_actions_official_scaled": gt.tolist(), "padding_mask": mask.tolist(),
            "action_columns": ["translation_x", "translation_y", "rotation_cos_component", "rotation_sin_component"],
            "official_direction_scale": 500,
        })
        write_json(args.output / "input_geometry.json", {
            "coordinate_scale_from_polygon_txt": 10,
            "official_max_contours": polygons,
        })
        polygon_copy_dir = args.output / "input_polygons"
        polygon_copy_dir.mkdir()
        for index in range(POLYGON_COUNT):
            shutil.copyfile(args.polygons / ("%d.txt" % index),
                            polygon_copy_dir / ("%d.txt" % index))
        torch.save(gt.clone(), args.output / "gt_input.pt")

        features = evaluation.create_gnn_data(polygons).to(device)
        _prior, marginal_prob, sde, _eps = init_sde("ve")
        model = PolygonPackingTransformer(
            marginal_prob_std_func=marginal_prob, device=device).to(device)
        load_options = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_options["weights_only"] = True
            metadata["checkpoint_load_mode"] = "weights_only=True"
        else:
            metadata["checkpoint_load_mode"] = "legacy torch.load; trusted official release only"
        state_dict = torch.load(args.checkpoint, **load_options)
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        model.eval()
        torch.cuda.synchronize(device)
        sample_started = time.perf_counter()
        with torch.no_grad():
            trajectory, prediction = pc_sampler_state(
                model, sde, ids.numel(), ids, features, mask,
                batch_size=args.candidates, num_steps=args.steps)
        torch.cuda.synchronize(device)
        metadata["sampler_seconds"] = time.perf_counter() - sample_started
        raw = prediction.detach().cpu().clone()
        trajectory_cpu = trajectory.detach().cpu()
        torch.save(raw, args.output / "predictions_raw.pt")
        torch.save(trajectory_cpu, args.output / "sampler_trajectory.pt")
        ensure_finite(torch, raw, "Raw prediction")
        ensure_finite(torch, trajectory_cpu, "Sampler trajectory")
        if tuple(raw.shape) != (args.candidates, ids.numel(), 4):
            raise ValueError("Sampler returned unexpected candidate dimensions")
        if tuple(trajectory_cpu.shape) != (args.candidates, args.steps, ids.numel(), 4):
            raise ValueError("Sampler returned unexpected trajectory dimensions")
        del trajectory, trajectory_cpu, prediction

        enhance_started = time.perf_counter()
        with torch.no_grad():
            official = evaluation.vali_res(ids, polygons, gt, mask, raw.clone())
        enhanced_with_gt = official[7].detach().cpu()
        torch.save(enhanced_with_gt[:-1].clone(), args.output / "predictions_enhanced.pt")
        torch.save(enhanced_with_gt[-1].clone(), args.output / "gt_enhanced.pt")
        ensure_finite(torch, enhanced_with_gt, "Enhanced candidates and GT")
        if tuple(enhanced_with_gt.shape) != (args.candidates + 1, ids.numel(), 4):
            raise ValueError("vali_res returned unexpected candidate/GT dimensions")
        metadata["enhancement_seconds"] = time.perf_counter() - enhance_started
        metadata["official_remove_spacing_seconds"] = float(official[10])

        metrics = {
            "raw": native_metrics(torch, evaluation, polygons, ids, raw),
            "enhanced": native_metrics(torch, evaluation, polygons, ids, enhanced_with_gt[:-1]),
            "gt_input": native_metrics(torch, evaluation, polygons, ids, gt.unsqueeze(0)),
            "gt_enhanced": native_metrics(torch, evaluation, polygons, ids, enhanced_with_gt[-1:]),
        }
        if (metrics["raw"]["official_valid_count"] != int(official[0]) or
                metrics["enhanced"]["official_valid_count"] != int(official[1])):
            raise ValueError("Independent official metric recomputation disagrees with vali_res")
        best_index = metrics["enhanced"]["best_official_valid_index"]
        preview_index = best_index if best_index is not None else 0
        metadata["preview_candidate_index"] = preview_index
        metadata["preview_is_official_valid"] = best_index is not None
        metadata["official_best_candidate_index_returned"] = int(official[8])
        metadata["preview_selection"] = "Highest official-valid utilization; index 0 fallback if none valid"
        for name, actions in (("raw_preview", raw[preview_index]),
                              ("enhanced_preview", enhanced_with_gt[preview_index]),
                              ("gt_input", gt), ("gt_enhanced", enhanced_with_gt[-1])):
            metrics[name + "_independent_geometry"] = exact_preview_audit(polygons, ids, actions)
            evaluation.visualize(0, ids, polygons, actions, mask,
                                 PNGWriter(args.output / (name + ".png")), name)
        write_json(args.output / "metrics.json", metrics)
        metadata["metric_summary"] = {
            name: {key: value[key] for key in
                   ("candidate_count", "official_valid_count", "best_official_valid_index",
                    "best_official_valid_utilization")}
            for name, value in metrics.items() if "candidate_count" in value}
        metadata["status"] = "completed"
        print("Completed %s: sample=%d candidates=%d steps=%d enhanced_official_valid=%d/%d" %
              (metadata["run_kind"], args.sample_index, args.candidates, args.steps,
               metrics["enhanced"]["official_valid_count"], args.candidates))
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        raise
    finally:
        metadata["finished_utc"] = utc_now()
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        metadata["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        metadata["nvidia_smi_after"] = command_snapshot([
            "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"])
        metadata["output_sha256"] = {
            str(path.relative_to(args.output)): sha256(path)
            for path in sorted(args.output.rglob("*")) if path.is_file()}
        write_json(args.output / "metadata.json", metadata)


def main(argv=None):
    try:
        run(parse_args(argv))
        return 0
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
