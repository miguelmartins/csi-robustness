import torch


def random_uniform_linf_attack(x, epsilon, clamp_min=0.0, clamp_max=1.0):
    """
    Random uniform noise within an L∞ ball of radius epsilon:
      x_adv = clamp(x + U[-epsilon, epsilon], [clamp_min, clamp_max])
    """
    noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = x + noise
    return x_adv.clamp(clamp_min, clamp_max)


@torch.no_grad()
def evaluate_under_random_noise(model, loader, device, attack_fn, class_dim=1):
    """
    attack_fn(x) -> x_adv
    """
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(x)
        logits = model(x_adv)
        pred = logits.argmax(dim=class_dim)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total
