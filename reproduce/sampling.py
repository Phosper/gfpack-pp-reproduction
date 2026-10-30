"""Section 5.2 Gaussian VE Euler-Maruyama generation with explicit late constraints."""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .collision_flow import integrate_collision_field


def sample(model, ids, features, geometry, batch_size=512, seed=3407, sigma_min=.1,
           constraint_integrator='em', constraint_threads=1, constraint_max_substeps=32):
    if constraint_integrator not in ('em', 'split-flow'):
        raise ValueError('Unknown constraint integrator')
    if constraint_threads < 1 or constraint_max_substeps < 1:
        raise ValueError('Constraint workers and substep limit must be positive')
    device = model.device
    generator = torch.Generator(device=device).manual_seed(seed)
    count = len(ids)
    states = torch.randn((batch_size, count, 4), device=device, generator=generator) * 1000.
    all_ids = ids.repeat(batch_size)
    mask = torch.zeros(batch_size * count, device=device)
    batch = torch.arange(batch_size, device=device).repeat_interleave(count)
    # Paper specifies 128 steps and t0=1, but not t_end/grid. This grid is declared.
    times = torch.linspace(1., .01, 129, device=device)
    trace = []
    history = [states.detach().cpu().clone()]
    with torch.no_grad(), ThreadPoolExecutor(max_workers=constraint_threads) as pool:
        encoded = model.geo_feature(features)
        for index in range(128):
            t = times[index]
            dt = times[index] - times[index + 1]
            batch_t = t.expand(batch_size)
            score = model(states.reshape(-1, 4), all_ids, mask, batch, batch_t,
                          features, encoded).reshape_as(states)
            sigma = sigma_min * (1000. / sigma_min)**float(t)
            g = sigma * math.sqrt(2 * math.log(1000. / sigma_min))
            correction = torch.zeros_like(states)
            pair_count = 0
            flow_reports = []
            started = time.perf_counter()
            if index >= 96 and constraint_integrator == 'em':
                poses = states.detach().cpu().double().numpy()
                if constraint_threads > 1:
                    _ = geometry.parts
                def evaluate(pose):
                    return geometry.oriented(pose).field(pose[:, :2])
                fields = list(pool.map(evaluate, poses)) if constraint_threads > 1 else list(map(evaluate, poses))
                forces = [row[0] for row in fields]
                pair_count = sum(row[2] for row in fields)
                correction[:, :, :2] = torch.as_tensor(np.asarray(forces), device=device,
                                                      dtype=states.dtype)
            noise = torch.randn(states.shape, device=device, generator=generator)
            states = states + g*g*dt*(score + correction) + g*torch.sqrt(dt)*noise
            if index >= 96 and constraint_integrator == 'split-flow':
                # Lie splitting keeps the neural EM proposal and its random
                # draws intact, then integrates the same area-weighted field
                # at the proposed rotations. This is a declared numerical
                # reconstruction choice, not the authors' confirmed scheme.
                poses = states.detach().cpu().double().numpy()
                duration = g*g*float(dt)
                if constraint_threads > 1:
                    _ = geometry.parts  # Prepare immutable decompositions once.
                def advance(pose):
                    return integrate_collision_field(
                        geometry, pose, duration, max_substeps=constraint_max_substeps,
                        storage_dtype=np.float32)
                advanced = list(pool.map(advance, poses)) if constraint_threads > 1 else list(map(advance, poses))
                states = torch.as_tensor(np.asarray([a[0] for a in advanced]), device=device, dtype=states.dtype)
                flow_reports = [a[1] for a in advanced]
                pair_count = sum(r['collision_pairs_before'] for r in flow_reports)
            if not torch.isfinite(states).all():
                raise FloatingPointError(f'Nonfinite state at generation step {index+1}')
            trace.append({'step': index + 1, 'time': float(t), 'dt': float(dt),
                          'model_training_flag': getattr(model, 'training', None),
                          'grad_enabled': torch.is_grad_enabled(),
                          'constraint_active': index >= 96, 'collision_pairs': pair_count if index >= 96 else None,
                          'collision_pairs_measured': index >= 96,
                          'constraint_integrator': constraint_integrator,
                          'constraint_flow_energy_measured': bool(flow_reports),
                          'constraint_force_max': (max(r['force_max_before'] for r in flow_reports)
                                                   if flow_reports else float(correction[:, :, :2].norm(dim=-1).max())),
                          'constraint_substeps_total': sum(r['substeps'] for r in flow_reports),
                          'constraint_substeps_max': max((r['substeps'] for r in flow_reports), default=0),
                          'constraint_unintegrated_time_max': max((r['unintegrated_time'] for r in flow_reports), default=0.),
                          'constraint_rejected_trials': sum(r['rejected_trials'] for r in flow_reports),
                          'constraint_energy_before_sum': sum(r['energy_before'] for r in flow_reports),
                          'constraint_energy_after_sum': sum(r['energy_after'] for r in flow_reports),
                          'constraint_max_substep_displacement': max((r['max_substep_displacement'] for r in flow_reports), default=0.),
                          'update_seconds': time.perf_counter() - started})
            history.append(states.detach().cpu().clone())
            if (index + 1) % 16 == 0:
                print(f'Generation {index+1}/128; constraints={index >= 96}', flush=True)
    return states.detach().cpu().double().numpy(), torch.stack(history, dim=1), trace
