#!/usr/bin/env python3
"""Paired, unweighted denoising-score diagnostic; never train or sample.

The fixed t grid is a diagnostic grid, not an estimator of the training-time
uniform-t expectation. Lower loss supports local compatibility only: the
optimal marginal score need not recover the particular conditional noise draw.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


TIMES = (0.01, 0.05, 0.15, 0.35, 0.60, 0.85, 1.0)
PHYSICAL_SIGMAS = (1.1, 3.0, 10.0, 100.0, 1000.0)
SAMPLES = (0, 45)
REPLICATES = 4
CONFIGURATIONS = tuple((s, r) for s in (1.0, 0.1) for r in (500.0, 1000.0))
SEED = 3407
STD_EPS = 1e-5


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sigma_at(t, sigma_min):
    return sigma_min * (1000.0 / sigma_min) ** t


def diagnostic_grid(sigma_min):
    rows = [{"grid_group": "fixed_t", "grid_value": t, "t": t,
             "physical_sigma": sigma_at(t, sigma_min)} for t in TIMES]
    for sigma in PHYSICAL_SIGMAS:
        t = math.log(sigma / sigma_min) / math.log(1000.0 / sigma_min)
        if not .01 <= t <= 1.:
            raise ValueError("Matched physical sigma lies outside the common t range")
        rows.append({"grid_group": "matched_sigma", "grid_value": sigma,
                     "t": t, "physical_sigma": sigma})
    return rows


def make_grid(torch, original_gt, sigma_min, rotation_scale, sample_index):
    """Generate common CPU noise independently of model/dropout RNG."""
    n = original_gt.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(SEED + sample_index)
    grid = diagnostic_grid(sigma_min)
    noise = torch.randn(len(grid) * REPLICATES, n, 4, generator=generator)
    t = torch.tensor([row["t"] for row in grid], dtype=torch.float32).repeat_interleave(REPLICATES)
    clean = original_gt.clone()
    clean[:, 2:] *= rotation_scale
    clean = clean.unsqueeze(0).expand(len(t), -1, -1).clone()
    effective_std = sigma_at(t, sigma_min) + STD_EPS
    # Set corruption amplitudes exactly equal in the matched-sigma group. The
    # model computes its denominator from float32 t, whose roundoff is reported.
    effective_std[len(TIMES) * REPLICATES:] = torch.tensor(
        PHYSICAL_SIGMAS, dtype=torch.float32).repeat_interleave(REPLICATES) + STD_EPS
    noisy = clean + effective_std[:, None, None] * noise
    return clean, noisy, noise, t, effective_std


def score_diagnostics(torch, clean, noisy, noise, score, effective_std, active):
    """All MSE means exclude masked polygons; four channels remain separate."""
    if not active.any().item():
        raise ValueError("No active polygons")
    if not torch.isfinite(score).all().item():
        raise FloatingPointError("Nonfinite checkpoint score")
    std = effective_std[:, None, None]
    delta = score * std + noise
    reconstruction = noisy + std.square() * score
    # Evaluating reconstruction independently also checks the algebra and units.
    normalized_error = (reconstruction - clean) / std
    mask = active[None, :, None].expand_as(delta)
    channel_mse = delta[:, active, :].square().mean(dim=1)
    baseline_mse = noise[:, active, :].square().mean(dim=1)
    reconstruction_mse = (reconstruction - clean)[:, active, :].square().mean(dim=1)
    normalized_reconstruction_mse = normalized_error[:, active, :].square().mean(dim=1)
    n = int(active.sum().item())
    if not torch.isfinite(delta[mask]).all().item():
        raise FloatingPointError("Nonfinite noise-normalized residual")
    gt_theta = torch.atan2(clean[:, active, 3], clean[:, active, 2])
    noisy_theta = torch.atan2(noisy[:, active, 3], noisy[:, active, 2])
    estimate_theta = torch.atan2(reconstruction[:, active, 3], reconstruction[:, active, 2])
    def angular_error(theta):
        error = torch.atan2(torch.sin(theta - gt_theta), torch.cos(theta - gt_theta))
        return error.abs().mean(dim=1) * (180.0 / math.pi)
    return {
        "channel_normalized_mse": channel_mse,
        "channel_zero_score_mse": baseline_mse,
        "channel_reconstruction_mse_original_units": reconstruction_mse,
        "channel_reconstruction_mse_divided_by_sigma_squared": normalized_reconstruction_mse,
        "unweighted_source_loss_per_placement": channel_mse.mean(dim=1) * n,
        "noisy_angle_mean_abs_error_degrees": angular_error(noisy_theta),
        "reconstruction_angle_mean_abs_error_degrees": angular_error(estimate_theta),
        "reconstruction_rotation_near_zero_count":
            (reconstruction[:, active, 2:].square().sum(dim=-1) <= 1e-12).sum(dim=1),
    }


def summarize_group(torch, values, indices):
    result = {key: value[indices].to(torch.float64).mean(dim=0).detach().cpu().tolist()
              for key, value in values.items()}
    error = values["channel_normalized_mse"][indices].sum(dim=0)
    baseline = values["channel_zero_score_mse"][indices].sum(dim=0)
    result["channel_ratio_to_paired_zero_score"] = (error / baseline).detach().cpu().tolist()
    result["translation_ratio_to_paired_zero_score"] = float(error[:2].sum() / baseline[:2].sum())
    result["rotation_ratio_to_paired_zero_score"] = float(error[2:].sum() / baseline[2:].sum())
    result["all_channel_ratio_to_paired_zero_score"] = float(error.sum() / baseline.sum())
    return result


def paired_forward(torch, model, noisy, ids, mask, t, features, seed):
    device = noisy.device
    devices = [device.index or 0] if device.type == "cuda" else []
    batch_size, n, _ = noisy.shape
    batch = torch.arange(batch_size, device=device).repeat_interleave(n)
    expanded_ids = ids.repeat(batch_size)
    expanded_mask = mask.repeat(batch_size)
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        before = {"cpu": tensor_digest(torch.get_rng_state())}
        if devices:
            before["cuda"] = tensor_digest(torch.cuda.get_rng_state(device))
        result = model(noisy.reshape(-1, 4), expanded_ids, expanded_mask,
                       batch, t, features).reshape(batch_size, n, 4)
        if devices:
            torch.cuda.synchronize(device)
        after = {"cpu": tensor_digest(torch.get_rng_state())}
        if devices:
            after["cuda"] = tensor_digest(torch.cuda.get_rng_state(device))
    return result, {"before": before, "after": after}


def self_test():
    import torch
    gt = torch.tensor([[4., 6., 1., 0.], [2., 3., 0., 1.], [0., 0., 1., 0.]])
    active = torch.tensor([True, True, False])
    checks = []
    noise_hashes = []
    matched_inputs = {}
    for sigma_min, rotation_scale in CONFIGURATIONS:
        clean, noisy, noise, t, std = make_grid(torch, gt, sigma_min, rotation_scale, 0)
        noise_hashes.append(tensor_digest(noise))
        matched_inputs[(sigma_min, rotation_scale)] = noisy[len(TIMES) * REPLICATES:].clone()
        assert torch.equal(clean[0, :, :2], gt[:, :2])
        assert torch.equal(clean[0, :, 2:], gt[:, 2:] * rotation_scale)
        assert torch.allclose(noisy, clean + std[:, None, None] * noise)
        zero = score_diagnostics(torch, clean, noisy, noise, torch.zeros_like(noise), std, active)
        z = summarize_group(torch, zero, slice(None))
        assert z["all_channel_ratio_to_paired_zero_score"] == 1.0
        oracle = score_diagnostics(torch, clean, noisy, noise, -noise / std[:, None, None], std, active)
        assert oracle["channel_normalized_mse"].max().item() < 1e-12
        assert torch.allclose(zero["unweighted_source_loss_per_placement"],
                              zero["channel_normalized_mse"].mean(dim=1) * 2)
        changed = torch.zeros_like(noise)
        changed[:, ~active, :] = 1e20
        masked = score_diagnostics(torch, clean, noisy, noise, changed, std, active)
        assert torch.equal(masked["channel_normalized_mse"], zero["channel_normalized_mse"])
    assert len(set(noise_hashes)) == 1
    for rotation_scale in (500., 1000.):
        assert torch.equal(matched_inputs[(1., rotation_scale)], matched_inputs[(.1, rotation_scale)])
    assert all(diagnostic_grid(s)[-1]["t"] == 1. for s in (1., .1))
    checks += ["four_config_input_scales", "same_epsilon_four_configurations",
               "matched_physical_sigma_inputs_equal_between_sigma_profiles",
               "sigma_1000_endpoint_common_t_one",
               "zero_score_paired_ratio_one", "conditional_oracle_residual_zero",
               "source_unweighted_loss_N_scaling", "masked_polygons_excluded"]
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = torch.nn.Dropout(.2)
        def forward(self, actions, ids, mask, batch, t, features):
            return self.drop(torch.ones_like(actions))
    fake = Fake().train()
    before = torch.get_rng_state().clone()
    outputs = []
    rng_records = []
    for sigma_min, rotation_scale in CONFIGURATIONS:
        _, noisy, _, t, _ = make_grid(torch, gt, sigma_min, rotation_scale, 0)
        output, record = paired_forward(torch, fake, noisy, torch.tensor([0, 1, 2]),
                                        torch.zeros(3), t, None, 29)
        outputs.append(output); rng_records.append(record)
        assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(outputs[0], x) for x in outputs[1:])
    assert all(rng_records[0] == x for x in rng_records[1:])
    checks += ["dropout_rng_paired_four_configurations", "caller_rng_restored"]
    plan = make_plan()
    assert len(plan) == 96 and sum(x["planned_replicates"] for x in plan) == 384
    plan[0]["status"] = "measured";plan[0]["measured_replicates"] = 4
    totals = coverage(plan)
    assert totals["planned_noisy_placements"] == 384 and totals["measured_noisy_placements"] == 4
    assert totals["unmeasured_noisy_placements"] == 380
    checks.append("partial_run_preserves_all_planned_denominators")
    return {"status": "passed", "passed": len(checks), "checks": checks,
            "oracle_note": "A synthetic conditional oracle tests arithmetic only; a true marginal score is not expected to make every conditional-noise residual zero."}


def make_plan():
    return [{"sigma_min": s, "rotation_feature_scale": r, "sample_index": index,
             **grid, "planned_replicates": REPLICATES, "measured_replicates": 0,
             "status": "unmeasured"}
            for s, r in CONFIGURATIONS for index in SAMPLES for grid in diagnostic_grid(s)]


def coverage(plan):
    planned = sum(row["planned_replicates"] for row in plan)
    measured = sum(row["measured_replicates"] for row in plan)
    return {"planned_grid_cells": len(plan),
            "measured_grid_cells": sum(row["status"] == "measured" for row in plan),
            "planned_noisy_placements": planned, "measured_noisy_placements": measured,
            "unmeasured_noisy_placements": planned - measured}


def write_new_json(path, data):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(data, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Existing upstream/resources/extensions experiment root")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside protected sources")
    parser.add_argument("--max-seconds", type=float, default=60., help="Soft total wall-time budget, at most 60; no new forward begins after deadline")
    parser.add_argument("--self-test", action="store_true", help="Fast CPU arithmetic/RNG controls only")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if not 0 < args.max_seconds <= 60:
        parser.error("--max-seconds must be in (0,60]")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Output already exists")
    if args.self_test:
        result = self_test();write_new_json(out, result);print(json.dumps(result));return
    if args.root is None:
        parser.error("--root is required without --self-test")
    started = time.perf_counter()
    root = args.root.resolve();upstream = root / "upstream"
    if upstream == out or upstream in out.parents or out in root.parents:
        raise ValueError("Output must be outside protected sources")
    checkpoint = root / "resources/checkpoints/score_model.pth"
    dataset = root / "resources/data/datasets/dataset_dental_vali_128.pkl"
    inputs = [checkpoint, dataset, Path(__file__).resolve(),
              Path(__file__).resolve().with_name("run_inference.py")]
    inputs += sorted((upstream / "src").glob("*.py"))
    inputs += sorted((root / "resources/data/polygons").glob("*.txt"))
    inputs += sorted((root / "extensions").glob("*.so"))
    def label(path):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return "probe_scripts/" + path.name
    frozen = {label(p): digest_file(p) for p in inputs}
    plan = make_plan()
    report = {"status": "running", "purpose": "unweighted local checkpoint compatibility diagnostic",
        "training_performed": False, "sampling_performed": False, "postprocessing_performed": False,
        "checkpoint_training_configuration_verified": False, "input_sha256": frozen,
        "configurations": [{"sigma_min": s, "sigma_max": 1000., "rotation_feature_scale": r}
                           for s, r in CONFIGURATIONS],
        "sample_indices": SAMPLES, "fixed_time_grid": TIMES, "matched_physical_sigma_grid": PHYSICAL_SIGMAS,
        "noise_replicates_per_grid_point": REPLICATES,
        "noise_seed_policy": "CPU generator seed3407+sample_index; identical epsilon in all four configurations",
        "dropout_policy": "model.train; full forward including geometry encoder repeated with paired RNG seed13407+sample_index; caller CPU/CUDA RNG restored",
        "coordinate_policy": "official maximal contour x10; original GT x/y untouched; cosine/sine multiplied by declared 500 or 1000, without normalizing GT directions",
        "source_loss_formula": "src/sde.py lossFun: sum_polygons_channels((score*(sigma+1e-5)+epsilon)^2)/4, then sample-utilization-weighted average",
        "std_cancellation_note": "score_model.forward divides its final raw network output by (std+1e-5); multiplying by the same std in the DSM residual cancels this explicit scale. Differences primarily reflect noisy inputs and time conditioning, with float32 denominator roundoff recorded; this does not identify training sigma.",
        "probe_loss_policy": "unweighted; training utilization weights unavailable; report per-element and N-scaled original-formula diagnostics separately",
        "normalization": "delta=score*(sigma+1e-5)+epsilon; channel MSE divided by same paired epsilon MSE for zero-score baseline ratio",
        "channel_order": ["x", "y", "scaled_cos", "scaled_sin"],
        "t_grid_policy": "fixed nonuniform diagnostic grid; equal-grid summaries are descriptive, not uniform-time training-loss estimates",
        "matched_sigma_policy": "same physical sigma, clean GT, and epsilon at each rotation scale; profile-specific t=log(sigma/sigma_min)/log(1000/sigma_min); group summaries kept separate",
        "matched_sigma_roundoff": "corruption sigma is exactly the common float32 physical sigma; model std is recomputed from float32 t; small denominator roundoff is recorded per forward",
        "tweedie_policy": "one-step diagnostic clean_estimate=noisy+(sigma+1e-5)^2*score; not a generative sampler",
        "interpretation_limits": [
            "Same t but different sigma_min changes corruption amplitude and task difficulty.",
            "Rotation scale changes clean input distribution, signal-to-noise ratio, and angle error interpretation.",
            "A marginal score minimizes expected conditional-noise loss; a real model is not expected to equal -epsilon/sigma for each draw.",
            "Two validation GTs, four fixed noises, and paired dropout do not identify checkpoint training settings, architecture provenance, or packing success.",
            "No training-utilization weighting, stochastic uniform-t integration, or confidence intervals are reconstructed.",
            "Tweedie original-unit errors across rotation scales are not directly comparable; use normalized channel residuals and paired baselines, still subject to distribution differences."],
        "max_seconds": args.max_seconds, "budget_policy": "soft wall-time deadline checked before each small forward; an in-flight CUDA forward may finish beyond deadline",
        "grid": plan, "batch_rng": [], "sigma_1000_endpoint_checks": []}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for checkpoint probe; use --self-test for CPU controls")
        torch.set_num_threads(16)
        torch.backends.cudnn.benchmark = False
        sys.path.insert(0, str(upstream))
        import src
        src.__path__.insert(0, str(root / "extensions"))
        from src import evaluate as evaluation
        from src.score_model import PolygonPackingTransformer
        from run_inference import checked_sample, command_snapshot
        if command_snapshot(["git", "rev-parse", "HEAD"], upstream).get("stdout") != "ff75815e24ec50cd7fa8b2abdec1acfe0b55be95":
            raise ValueError("Unexpected upstream revision")
        if command_snapshot(["git", "status", "--porcelain"], upstream).get("stdout"):
            raise ValueError("Upstream must remain clean")
        evaluation.POLYGON_DIR = root / "resources/data/polygons"
        polygons = evaluation.readAllPolys(440)
        device = torch.device("cuda", 0)
        features = evaluation.create_gnn_data(polygons).to(device)
        selected = {i: checked_sample(torch, evaluation, dataset, i) for i in SAMPLES}
        def std_fn(x, t):
            return x, sigma_at(t, 1.0)
        with torch.random.fork_rng(devices=[0]):
            model = PolygonPackingTransformer(marginal_prob_std_func=std_fn, device=device).to(device)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            report["state_dict_elements"] = sum(v.numel() for v in state.values())
            del state
        model.train()
        model.requires_grad_(False)
        endpoint_scores = {}
        report["model_parameters"] = sum(p.numel() for p in model.parameters())
        report["software"] = {"pytorch": str(torch.__version__), "cuda": torch.version.cuda}
        torch.cuda.reset_peak_memory_stats(device)
        for sigma_min, rotation_scale in CONFIGURATIONS:
            model.marginal_prob_std = lambda x, t, s=sigma_min: (x, sigma_at(t, s))
            for index in SAMPLES:
                if time.perf_counter() - started >= args.max_seconds:
                    raise TimeoutError("Soft wall-time budget reached before next forward")
                ids, original, _, mask, _ = selected[index]
                clean, noisy, noise, t, std = make_grid(torch, original, sigma_min, rotation_scale, index)
                active = mask == 0
                eps_hash = tensor_digest(noise)
                batch_started = time.perf_counter()
                score, rng = paired_forward(torch, model, noisy.to(device), ids.to(device), mask.to(device),
                                            t.to(device), features, SEED + 10000 + index)
                endpoint_scores[(sigma_min, rotation_scale, index)] = score[-REPLICATES:].detach().cpu().clone()
                values = score_diagnostics(torch, clean.to(device), noisy.to(device), noise.to(device),
                                           score, std.to(device), active.to(device))
                split = len(TIMES) * REPLICATES
                group_indices = {"fixed_t": slice(0, split), "matched_sigma": slice(split, None)}
                record = {"sigma_min": sigma_min, "rotation_feature_scale": rotation_scale,
                          "sample_index": index, "noise_sha256": eps_hash, "rng": rng,
                          "matched_sigma_input_sha256": tensor_digest(noisy[split:]),
                          "max_relative_model_std_roundoff": float(((sigma_at(t.to(device), sigma_min) + STD_EPS - std.to(device)).abs() / std.to(device)).max()),
                          "elapsed_seconds": time.perf_counter() - batch_started,
                          "unweighted_grid_summary": {name: summarize_group(torch, values, indices)
                              for name, indices in group_indices.items()}}
                report["batch_rng"].append(record)
                for time_index, point in enumerate(diagnostic_grid(sigma_min)):
                    row = next(row for row in plan if row["sigma_min"] == sigma_min
                        and row["rotation_feature_scale"] == rotation_scale
                        and row["sample_index"] == index and row["grid_group"] == point["grid_group"]
                        and row["grid_value"] == point["grid_value"])
                    indices = slice(time_index * REPLICATES, (time_index + 1) * REPLICATES)
                    row.update({"status": "measured", "measured_replicates": REPLICATES,
                                "sigma_before_epsilon": point["physical_sigma"],
                                "active_polygons": int(active.sum()), "metrics": summarize_group(torch, values, indices)})
                print(json.dumps({"sigma_min": sigma_min, "rotation_feature_scale": rotation_scale,
                                  "sample_index": index, "elapsed_seconds": time.perf_counter() - started}), flush=True)
                del score, values
        for index in SAMPLES:
            rows = [row for row in report["batch_rng"] if row["sample_index"] == index]
            if len({row["noise_sha256"] for row in rows}) != 1 or len({json.dumps(row["rng"], sort_keys=True) for row in rows}) != 1:
                raise AssertionError("Paired noise/dropout RNG did not match across configurations")
            for rotation_scale in (500., 1000.):
                paired = [row for row in rows if row["rotation_feature_scale"] == rotation_scale]
                if len({row["matched_sigma_input_sha256"] for row in paired}) != 1:
                    raise AssertionError("Matched physical sigma inputs differ")
        report["noise_and_dropout_pairing_verified"] = True
        for rotation_scale in (500., 1000.):
            for index in SAMPLES:
                a = endpoint_scores[(1., rotation_scale, index)]
                b = endpoint_scores[(.1, rotation_scale, index)]
                close = bool(torch.allclose(a, b, rtol=1e-5, atol=1e-7))
                report["sigma_1000_endpoint_checks"].append({
                    "rotation_feature_scale": rotation_scale, "sample_index": index,
                    "physical_sigma": 1000., "both_t": 1., "rtol": 1e-5, "atol": 1e-7,
                    "max_absolute_score_difference": float((a - b).abs().max()), "passed": close})
                if not close:
                    raise AssertionError("Common sigma1000/t1 endpoint scores did not match")
        report["sigma_1000_endpoint_pairing_verified"] = True
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "partial" if any(row["status"] == "measured" for row in plan) else "failed"
        message = str(exc).replace(str(root), "<ROOT>").replace(str(Path(__file__).resolve().parent), "<SCRIPTS>")
        report["error"] = {"type": type(exc).__name__, "message": message}
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["coverage"] = coverage(plan)
        report["changed_inputs"] = [label(p) for p in inputs if digest_file(p) != frozen[label(p)]]
        if report["changed_inputs"]:
            report["status"] = "input_integrity_failed"
        write_new_json(out, report)
    print(json.dumps({"status": report["status"], "coverage": report["coverage"], "elapsed_seconds": report["elapsed_seconds"]}))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
