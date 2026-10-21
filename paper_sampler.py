"""Isolated GFPack++ sampling profiles; never modifies the released sampler.

``paper_settings`` implements the numerical settings in arXiv:2406.07579v2,
Sections 4.3--4.4 and 5.2, with the released network supplied by the caller.
Changing its marginal standard deviation does not establish that the released
checkpoint was trained with those settings. ``paper_mechanism_release_scale``
uses the same sampling mechanism with the release's marginal sigma/input scale.
``release_weights_constraints``
keeps the pinned release's sampling arithmetic and random draw order and adds
an optional constraint term. Neither profile silently repairs failed states.
"""

import math
import time

import torch


PROFILES = ("paper_settings", "paper_mechanism_release_scale", "release_weights_constraints")
NUM_STEPS = 128
CONSTRAINT_START_STEP = 97


def profile_settings(profile="paper_settings"):
    """Return plain metadata; the runner must construct a matching model std.

    ``rotation_feature_scale`` describes GT/input encoding, not a projection
    of generated cosine/sine components onto a circle of that radius.
    """
    if profile not in PROFILES:
        raise ValueError("Unknown sampling profile: %s" % profile)
    paper_scale = profile == "paper_settings"
    paper_mechanism = profile != "release_weights_constraints"
    return {
        "profile": profile,
        "sigma_min": 0.1 if paper_scale else 1.0,
        "sigma_max": 1000.0,
        "rotation_feature_scale": 1000.0 if paper_scale else 500.0,
        "rotation_norm_projection": False,
        "num_steps": NUM_STEPS,
        "t_start": 1.0,
        "t_end": 0.01,
        "time_grid": ("128 equal intervals, left endpoint score evaluations"
                      if paper_mechanism else "128 linspace points; repeated last interval"),
        "time_grid_provenance": (
            "Implementation assumption: the paper gives 128 Euler-Maruyama steps "
            "and t0=1 but does not fix the generation endpoint or grid. We use "
            "128 equal intervals from 1 to 0.01 (its training-time lower limit)."
            if paper_mechanism else "Exact time-grid convention from pinned src.sde.pc_sampler_state"
        ),
        "base_step_size": 0.99 / (NUM_STEPS if paper_mechanism else NUM_STEPS - 1),
        "drift_step_multiplier": 1.0 if paper_mechanism else 14.0,
        "per_step_uniform_jitter_width": 0.0 if paper_mechanism else 0.001,
        "initialization": ("Gaussian mean 0, standard deviation 1000, all 4 columns"
                           if paper_mechanism else "Uniform [0, 1), all 4 columns"),
        "constraint_first_step_1based": CONSTRAINT_START_STEP,
        "constraint_last_step_1based": NUM_STEPS,
        "constraint_coefficient": 1.0,
        "constraint_callback_semantics": "repulsion = negative collision-potential gradient proxy",
        "constraint_update": "translation += g(t)^2 * effective_positive_dt * repulsion",
        "constraint_input": "pre-update actions, detached CPU clone, shape (batch, polygons, 4)",
        "constraint_output": "repulsion in unchanged translation units, shape (batch, polygons, 2)",
        "constraint_unit_conversion": "none; no division by rotation-feature scale",
        "constraint_clipping": False,
        "constraint_sign_convention_note": (
            "Eq. 10 with decreasing diffusion time uses positive reverse dt. "
            "Callback returns -grad(Phi), so the correction is added. The paper's "
            "description of the separating displacement does not independently "
            "specify its sign; this implementation declares the descent convention."
        ),
        "last_step_includes_noise": True,
        "checkpoint_training_settings_verified": False,
        "checkpoint_note": (
            "Released weights are retained. The requested profile is a configuration "
            "probe, not evidence of matched training settings or full reproduction."
        ),
    }


def marginal_prob(profile="paper_settings"):
    """Build the VE mean/std callable accepted by the released score model."""
    settings = profile_settings(profile)
    sigma_min, sigma_max = settings["sigma_min"], settings["sigma_max"]

    def ve_mean_std(x, t):
        return x, sigma_min * (sigma_max / sigma_min) ** t

    return ve_mean_std


def sde_coeff(profile="paper_settings"):
    """Build the VE (zero drift, diffusion) callable for a named profile."""
    settings = profile_settings(profile)
    sigma_min, sigma_max = settings["sigma_min"], settings["sigma_max"]

    def ve_coeff(t):
        sigma = sigma_min * (sigma_max / sigma_min) ** t
        factor = torch.tensor(2 * (math.log(sigma_max) - math.log(sigma_min)), device=t.device)
        return torch.tensor(0, device=t.device), sigma * torch.sqrt(factor)

    return ve_coeff


class SamplingFailure(FloatingPointError):
    """A nonfinite state or constraint; completed step diagnostics are retained."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _finite(value, name, step, diagnostics):
    if not torch.isfinite(value).all().item():
        diagnostics["status"] = "failed_nonfinite"
        diagnostics["failure_step_1based"] = step
        diagnostics["failure_value"] = name
        raise SamplingFailure("%s contains nonfinite values at sampling step %s" %
                              (name, step), diagnostics)


def _device_of(model):
    if getattr(model, "device", None) is not None:
        return torch.device(model.device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def sample_paper(model, ids, features, mask, batch_size=512,
                 collision_callback=None, profile="paper_settings", seed=None,
                 *, num_steps=NUM_STEPS):
    """Return ``(trajectory, prediction, diagnostics)`` using the given profile.

    ``model`` uses the released seven-argument forward interface and exposes
    ``geo_feature(features)``. Its marginal/std callable and train/eval mode are
    the caller's responsibility. ``ids`` and ``mask`` each describe one sample;
    mask values are the official 0 (active) / -inf (padding) convention.

    The optional callback is evaluated on steps 97--128 on the current actions,
    before the stochastic update. It receives a CPU clone and returns a tensor
    or array containing *outward* translation repulsion (B, N, 2). It must not
    return already integrated displacements: this function applies g^2 * dt.
    Rotation corrections are identically zero. All callback values must be
    finite; no clipping or post-update projection hides instability.

    A supplied seed uses a temporary CPU/current-device RNG context, including
    dropout, without changing the caller's RNG sequence. With seed=None, draws
    use the caller's RNG, permitting release-profile arithmetic regression.
    Exactly 128 steps are required for the named profiles.
    """
    settings = profile_settings(profile)
    if isinstance(num_steps, bool) or num_steps != NUM_STEPS:
        raise ValueError("Named sampling profiles require exactly 128 steps")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None")
    if collision_callback is not None and not callable(collision_callback):
        raise TypeError("collision_callback must be callable or None")

    device = _device_of(model)
    input_ids = torch.as_tensor(ids, device=device).detach()
    if input_ids.ndim != 1 or input_ids.numel() < 1:
        raise ValueError("ids must be a nonempty one-dimensional sequence")
    if not torch.isfinite(input_ids).all().item():
        raise ValueError("ids contains nonfinite values")
    integer_ids = input_ids.to(dtype=torch.int64)
    if not torch.equal(input_ids, integer_ids.to(dtype=input_ids.dtype)):
        raise ValueError("ids must contain integers")
    if (integer_ids < 0).any().item():
        raise ValueError("ids must be nonnegative")
    padding = torch.as_tensor(mask, device=device, dtype=torch.float32).detach()
    if padding.ndim != 1 or padding.shape != input_ids.shape:
        raise ValueError("mask must have the same one-dimensional shape as ids")
    if not ((padding == 0) | torch.isneginf(padding)).all().item():
        raise ValueError("mask must use the official 0 / negative-infinity convention")
    if not (padding == 0).any().item():
        raise ValueError("mask must contain at least one active polygon")

    poly_count = integer_ids.numel()
    flat_ids = integer_ids.unsqueeze(0).repeat(batch_size, 1).reshape(-1)
    flat_mask = padding.unsqueeze(0).repeat(batch_size, 1).reshape(-1)
    flat_batch = torch.arange(batch_size, device=device, dtype=torch.int64).repeat_interleave(poly_count)
    active = (flat_mask == 0).to(dtype=torch.float32).reshape(-1, 1)
    model_ids = (flat_ids.reshape(-1, 1) * active).to(dtype=torch.int64).reshape(-1)
    paper = profile != "release_weights_constraints"
    diagnostics = dict(settings)
    diagnostics.update({
        "status": "running", "batch_size": batch_size, "polygon_count": poly_count,
        "active_polygon_count": int((padding == 0).sum().item()),
        "seed": seed, "device": str(device),
        "model_training_flag": bool(getattr(model, "training", False)),
        "constraints_enabled": collision_callback is not None,
        "constraint_callback_calls": 0, "constraint_callback_seconds": 0.0,
        "marginal_std_setup": "Caller must construct model with this profile's marginal_prob callable",
        "paper_algorithm_complete": False,
        "paper_algorithm_complete_note": "Sampling configuration only; checkpoint match and enhancement are separate",
        "steps": [],
    })
    rng_devices = []
    if device.type == "cuda":
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=rng_devices, enabled=seed is not None):
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
            with torch.no_grad():
                shape = (batch_size, poly_count, 4)
                # Keep initialization before geo_feature to preserve release RNG order.
                initial = (torch.randn(shape, device=device) * settings["sigma_max"]
                           if paper else torch.rand(shape, device=device))
                state = initial.reshape(-1, 4)
                if paper:
                    time_edges = torch.linspace(1.0, 0.01, NUM_STEPS + 1, device=device)
                    time_steps = time_edges[:-1]
                    stochastic_dt = torch.tensor(settings["base_step_size"], device=device)
                else:
                    time_steps = torch.linspace(1.0, 0.01, NUM_STEPS, device=device)
                    stochastic_dt = time_steps[0] - time_steps[1]
                feature = model.geo_feature(features)
                trajectory = []
                for index, timestep in enumerate(time_steps):
                    step = index + 1
                    if paper:
                        effective_dt = stochastic_dt
                    else:
                        interval = (time_steps[index] - time_steps[index + 1]
                                    if index + 1 < NUM_STEPS else time_steps[index - 1] - time_steps[index])
                        effective_dt = interval * 14
                    base = state.clone() * active
                    if not paper:
                        state = state + (torch.rand_like(state) - 0.5) * 0.001
                    state = state * active
                    batch_t = torch.ones(batch_size, device=device) * timestep
                    score = model(state, model_ids, flat_mask, flat_batch, batch_t, features, feature)
                    if tuple(score.shape) != (batch_size * poly_count, 4):
                        raise ValueError("model returned unexpected score shape at step %d: %s" %
                                         (step, tuple(score.shape)))
                    score = score * active
                    _finite(score, "model score", step, diagnostics)

                    # Calculate g in the same order/precision as pinned src.sde.ve_sde.
                    sde_t = torch.ones(batch_size, device=device).unsqueeze(-1) * timestep
                    sigma = settings["sigma_min"] * (settings["sigma_max"] / settings["sigma_min"]) ** sde_t
                    log_ratio_twice = torch.tensor(2 * (math.log(settings["sigma_max"]) -
                                                       math.log(settings["sigma_min"])), device=device)
                    diffusion = sigma * torch.sqrt(log_ratio_twice)
                    diffusion = diffusion.view(-1, 1).repeat(1, poly_count).reshape(-1, 1)
                    drift = diffusion ** 2 * score
                    repulsion_max_norm = 0.0
                    correction_max_norm = 0.0
                    callback_seconds = 0.0
                    constrained = collision_callback is not None and step >= CONSTRAINT_START_STEP
                    if constrained:
                        callback_started = time.perf_counter()
                        repulsion_result = collision_callback(base.reshape(shape).detach().cpu().clone())
                        callback_seconds = time.perf_counter() - callback_started
                        repulsion = torch.as_tensor(repulsion_result, device=device, dtype=base.dtype).detach()
                        if tuple(repulsion.shape) != (batch_size, poly_count, 2):
                            raise ValueError("collision_callback must return (batch, polygons, 2); got %s at step %d" %
                                             (tuple(repulsion.shape), step))
                        _finite(repulsion, "collision repulsion", step, diagnostics)
                        repulsion = repulsion.reshape(-1, 2) * active
                        correction = torch.zeros_like(base)
                        correction[:, :2] = repulsion
                        drift = drift + diffusion ** 2 * correction
                        repulsion_max_norm = float(torch.linalg.vector_norm(repulsion, dim=-1).max().item())
                        correction_max_norm = float(torch.linalg.vector_norm(
                            (diffusion ** 2 * correction[:, :2]) * effective_dt, dim=-1).max().item())
                        diagnostics["constraint_callback_calls"] += 1
                        diagnostics["constraint_callback_seconds"] += callback_seconds

                    # Release drift multiplier deliberately does not scale its noise dt.
                    # The pinned code adds noise at all 128 steps despite its last-step comment.
                    delta = drift * effective_dt
                    state = base + delta + diffusion * torch.sqrt(stochastic_dt) * torch.randn_like(base)
                    # Padded rows do not represent polygons; keep them inert for both profiles.
                    state = state * active
                    _finite(state, "updated actions", step, diagnostics)
                    trajectory.append(state.clone().reshape(shape).unsqueeze(1))
                    diagnostics["steps"].append({
                        "step_1based": step, "t": float(timestep.item()),
                        "sigma": float(sigma[0, 0].item()),
                        "effective_drift_dt": float(effective_dt.item()),
                        "stochastic_dt": float(stochastic_dt.item()),
                        "g_squared_dt": float((diffusion[0, 0] ** 2 * effective_dt).item()),
                        "score_max_abs": float(score.abs().max().item()),
                        "state_max_abs": float(state.abs().max().item()),
                        "constraint_active": constrained,
                        "repulsion_max_norm": repulsion_max_norm,
                        "constraint_displacement_max_norm": correction_max_norm,
                        "callback_seconds": callback_seconds,
                    })
                result = torch.cat(trajectory, dim=1)
                prediction = state.reshape(shape)
                diagnostics["status"] = "completed"
                return result, prediction, diagnostics
    except Exception as exc:
        if diagnostics["status"] == "running":
            diagnostics["status"] = "failed"
            diagnostics["failure_step_1based"] = len(diagnostics["steps"]) + 1
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["error_message"] = str(exc)
        # Unexpected callback/model errors also retain diagnostics for the caller.
        if not hasattr(exc, "diagnostics"):
            try:
                exc.diagnostics = diagnostics
            except (AttributeError, TypeError):
                pass
        raise
    finally:
        diagnostics["elapsed_seconds"] = time.perf_counter() - started
