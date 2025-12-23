import torch


class UniformLinfAttackNoClamp:
    # We do not use clamp because it is tricky on binary images
    def __init__(self, eps: float, p: float = 1.0):
        self.eps = eps
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x expected in [0,1], shape [C,H,W]
        if self.p < 1.0 and torch.rand(()) > self.p:
            return x
        noise = torch.empty_like(x).uniform_(-self.eps, self.eps)
        return x + noise
