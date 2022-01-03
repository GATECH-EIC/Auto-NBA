import torch

def Random_alpha(model, epsilon):
    alpha = getattr(model.module, "alpha")
    alpha.data.add_(torch.zeros_like(alpha).uniform_(-epsilon, epsilon))
    model.module.clip()