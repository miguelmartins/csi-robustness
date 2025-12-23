import math
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
