"""Section 5.2 Gaussian VE Euler-Maruyama generation with explicit late constraints."""
import math
import time

import numpy as np
import torch


def sample(model, ids, features, geometry, batch_size=512, seed=3407, sigma_min=.1):
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
    with torch.no_grad():
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
            started = time.perf_counter()
            if index >= 96:
                poses = states.detach().cpu().double().numpy()
                forces = []
                for pose in poses:
                    force, _, pairs = geometry.oriented(pose).field(pose[:, :2])
                    forces.append(force)
                    pair_count += pairs
                correction[:, :, :2] = torch.as_tensor(np.asarray(forces), device=device,
                                                      dtype=states.dtype)
            noise = torch.randn(states.shape, device=device, generator=generator)
            states = states + g*g*dt*(score + correction) + g*torch.sqrt(dt)*noise
            if not torch.isfinite(states).all():
                raise FloatingPointError(f'Nonfinite state at generation step {index+1}')
            trace.append({'step': index + 1, 'time': float(t), 'dt': float(dt),
                          'model_training_flag': getattr(model, 'training', None),
                          'grad_enabled': torch.is_grad_enabled(),
                          'constraint_active': index >= 96, 'collision_pairs': pair_count,
                          'constraint_force_max': float(correction[:, :, :2].norm(dim=-1).max()),
                          'update_seconds': time.perf_counter() - started})
            history.append(states.detach().cpu().clone())
            if (index + 1) % 16 == 0:
                print(f'Generation {index+1}/128; constraints={index >= 96}', flush=True)
    return states.detach().cpu().double().numpy(), torch.stack(history, dim=1), trace
