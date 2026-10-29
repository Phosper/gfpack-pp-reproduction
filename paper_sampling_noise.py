"""Explicit common Gaussian draws for generation diagnostics, not a new sampler."""
import hashlib
import torch


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_noise_plan(batch_size, polygon_count, seed, steps=128):
    if any(type(v) is not int or v<1 for v in (batch_size,polygon_count,steps)) or type(seed) is not int:
        raise ValueError("Invalid noise dimensions/seed")
    generator=torch.Generator(device="cpu").manual_seed(seed)
    return {"seed":seed,
        "initial_standard_normal":torch.randn((batch_size,polygon_count,4),generator=generator),
        "brownian_standard_normal":torch.randn((steps,batch_size,polygon_count,4),generator=generator)}


def validate_noise_plan(plan, shape, steps=128):
    expected={"initial_standard_normal":tuple(shape),"brownian_standard_normal":(steps,*shape)}
    for name,dims in expected.items():
        value=plan[name]
        if (not isinstance(value,torch.Tensor) or value.device.type!="cpu" or value.dtype!=torch.float32 or
                tuple(value.shape)!=dims or not torch.isfinite(value).all()):
            raise ValueError("Invalid common-noise tensor: "+name)
    return {name:tensor_digest(plan[name]) for name in expected}
