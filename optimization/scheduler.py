import math
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer_and_scheduler(model, total_epochs, is_transformer=False):
    if is_transformer:
        lr = 2e-4
        wd = 0.01
    else:
        lr = 1e-3
        wd = 0.05
    params = [p for m in model for p in m.parameters()]
    optimizer = AdamW(params, lr=lr, weight_decay=wd)

    WARMUP_EPOCHS = 10

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            # linear warmup
            return float(epoch + 1) / WARMUP_EPOCHS
        else:
            # cosine
            progress = (epoch - WARMUP_EPOCHS) / max(1, total_epochs - WARMUP_EPOCHS)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


class LR_Scheduler(object):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        warmup_lr,
        num_epochs,
        base_lr,
        final_lr,
        iter_per_epoch,
        constant_predictor_lr=False,
    ):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group["name"] == "predictor":
                param_group["lr"] = self.base_lr
            else:
                lr = param_group["lr"] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def build_optimizer_and_scheduler_siamsiam(model, total_epochs, train_loader):
    lr = 0.03 * 512 / 256
    predictor_prefix = ("module.predictor", "predictor")
    parameters = [
        {
            "name": "base",
            "params": [
                param
                for name, param in model.named_parameters()
                if not name.startswith(predictor_prefix)
            ],
            "lr": lr,
        },
        {
            "name": "predictor",
            "params": [
                param
                for name, param in model.named_parameters()
                if name.startswith(predictor_prefix)
            ],
            "lr": lr,
        },
    ]
    lr = 0.03
    momentum = 0.9
    weight_decay = 0.0005
    final_lr = 0.0
    optimizer = torch.optim.SGD(
        parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    warmup_epochs = 10
    warmup_lr = 0
    batch_size = 512
    scheduler = LR_Scheduler(
        optimizer,
        warmup_epochs,
        warmup_lr * batch_size / 256,
        total_epochs,
        lr * batch_size / 256,
        final_lr * batch_size / 256,  # FIXED: Removed the leading period
        len(train_loader),  # FIXED: train_loader is now passed as an argument
        constant_predictor_lr=True,  # see the end of section 4.2 predictor
    )
    return optimizer, scheduler
