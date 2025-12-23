import torch
import torch.nn as nn
from torchvision.models import resnet18


def get_model(model, nc, out_size, device, seed):
    torch.manual_seed(seed)
    if model == "image":
        net = nn.Sequential(
            nn.Flatten(1),
        ).to(device)
    elif model == "linear":
        net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(nc * 64 * 64, 512),
        ).to(device)
    elif model == "mlp":
        net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(nc * 64 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        ).to(device)
    elif model == "cnn":
        net = resnet18()
        net.conv1 = nn.Conv2d(nc, 64, 7, 2, 3, bias=False)
        net.fc = nn.Identity()
        net = net.to(device)
    else:
        raise ValueError("Model %s undefined." % model)
    return net
