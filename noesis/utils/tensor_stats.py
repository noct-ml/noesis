import torch

def quick_stats(t: torch.Tensor):
    x = t.detach()
    return dict(
        shape=list(x.shape),
        mean=float(x.float().mean().cpu()),
        std=float(x.float().std().cpu()),
        abs_mean=float(x.abs().float().mean().cpu()),
        max=float(x.float().amax().cpu()),
        min=float(x.float().amin().cpu()),
        dtype=str(x.dtype).replace("torch.", ""),
        device=str(x.device),
    )
