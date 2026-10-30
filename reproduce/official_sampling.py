"""Call the public author's sampler unchanged, with truthful callback traces.

This profile deliberately keeps the public uniform prior, drift multiplier 14,
tiny score-input perturbation and missing late geometric callback. It is a
source baseline, not the paper's explicitly stated numerical configuration.
"""
import torch

from src.sde import init_sde, pc_sampler_state


def sample_public(model, ids, features, batch_size=512):
    device = model.device
    # Observe the initial draw without consuming additional RNG relative to
    # the unmodified public sampler.
    if device.type == 'cuda':
        rng = torch.cuda.get_rng_state(device)
    else:
        rng = torch.get_rng_state()
    initial = torch.rand((batch_size, len(ids), 4), device=device)
    if device.type == 'cuda':
        torch.cuda.set_rng_state(rng, device)
    else:
        torch.set_rng_state(rng)
    times = torch.linspace(1., .01, 128, device=device)
    trace = []
    def observe(module, arguments):
        index = len(trace)
        time = arguments[4]
        dt = times[index]-times[index+1] if index < 127 else times[-2]-times[-1]
        trace.append({'step': index+1, 'time': float(time[0]), 'dt': float(dt),
                      'model_training_flag': module.training, 'grad_enabled': torch.is_grad_enabled(),
                      'constraint_active': False, 'constraint_integrator': 'none-public-source',
                      'collision_pairs': None, 'collision_pairs_measured': False,
                      'constraint_force_max': 0., 'public_drift_multiplier': 14.})
        if (index+1)%16==0:
            print(f'Public generation {index+1}/128; constraints=False',flush=True)
    handle = model.register_forward_pre_hook(observe)
    try:
        _, _, coefficient, _ = init_sde('ve')
        sequence, final = pc_sampler_state(model, coefficient, len(ids), ids, features,
                                           torch.zeros(len(ids), device=device), batch_size=batch_size,
                                           num_steps=128)
    finally:
        handle.remove()
    if len(trace) != 128:
        raise ValueError('Public sampler did not execute 128 neural updates')
    history = torch.cat([initial.detach().cpu().unsqueeze(1), sequence.detach().cpu()], dim=1)
    return final.detach().cpu().double().numpy(), history, trace
