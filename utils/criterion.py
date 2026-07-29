import torch


def l2_norm(a, b, reduce=True):
    if reduce:
        return torch.linalg.norm((a - b).reshape(-1), dim=0, ord=2)
    else:
        return torch.linalg.norm(a - b, dim=-1, ord=2, keepdim=True)
