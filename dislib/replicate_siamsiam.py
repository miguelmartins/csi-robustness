# 1: Implement color augs with gaussian blur
# 2: Implement Diet strength augs
# 3. See how they interact with random noise attack
# 4. See if we can have bigger batch size for diet
# 5: run all combinations
import argparse
import dislib.defaults as defaults
import numpy as np
import os
import torch
import torch.nn as nn

from dataset_processing.augmentations import (
    dsprites_augmentations,
    shapes3d_augmentations,
)
from dataset_processing.load_datasets import (
    ContrastiveDataset,
    ContrastiveDislibDataset,
    DietGrayDataset,
    DietMPI3DDataset,
    DietRGBDataset,
    DislibDataset,
    MPI3DDataset,
    RGBContrastiveDataset,
    RGBDataset,
)
from tqdm.auto import tqdm

from evaluation.identifiability import log_test_evaluation, log_validation
from evaluation.logging import Args, setup_logging
from models.baselines import get_model
from torchvision.transforms import v2

from optimization.scheduler import (
    build_optimizer_and_scheduler,
    build_optimizer_and_scheduler_siamsiam,
)
from dataset_processing.load_datasets import DietDataset
from evaluation.identifiability import evaluate
from dataset_processing.load_datasets import GrayDataset
from models.baselines import SimSiam


def train(args, dataset, device, log_file):
    args.batch_size = 512
    with open(log_file, "a") as file:
        print("\n\nTraining:", file=file)
    (
        train_dataloader,
        _,
        _,
        data,
        out_size,
        nc,
        cat_ind,
    ) = dataset

    backbone = get_model(args.model, nc, out_size, device, args.seed)
    net = SimSiam(backbone=backbone).to(device)
    net.projector.set_layers(2)

    model = torch.nn.DataParallel(net)
    optimizer, scheduler = build_optimizer_and_scheduler_siamsiam(
        net, args.num_epochs, train_dataloader
    )
    global_progress = tqdm(range(0, args.num_epochs), desc=f"Training")
    for epoch in global_progress:
        model.train()

        local_progress = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}/{args.num_epochs}",
            disable=False,
        )
        for idx, (images1, images2, _) in enumerate(local_progress):
            model.zero_grad()
            data_dict = model.forward(
                images1.to(device, non_blocking=True),
                images2.to(device, non_blocking=True),
            )
            loss = data_dict["loss"].mean()  # ddp
            loss.backward()
            optimizer.step()
            scheduler.step()
            data_dict.update({"lr": scheduler.get_lr()})

            local_progress.set_postfix(data_dict)

        epoch_dict = {"epoch": epoch}
        global_progress.set_postfix(epoch_dict)
    torch.save(model.module.state_dict(), os.path.join(args.log_dir, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script LinRep")
    parser.add_argument("--rep", type=int, default=0, help="repetetion")
    parser.add_argument(
        "--backbone", type=str, default="cnn", help="mlp or cnn (resnet18)"
    )
    parser.add_argument(
        "--aug", type=str, default="none", help="Augmentations in train"
    )
    parser.add_argument("--dataset", type=str, default="dsprites", help="Dataset")

    rep = parser.parse_args().rep
    backbone = parser.parse_args().backbone
    aug = parser.parse_args().aug
    dataset = parser.parse_args().dataset

    settings = []
    print("Running setting:", "rep:", rep, "dataset:", dataset, "backbone:", backbone)

    args = Args()
    args.seed = defaults.SEED + rep
    args.dataset = dataset
    args.model = backbone
    args.batch_size = 512
    args.num_epochs = 100
    args.log_dir = os.path.join(
        defaults.SAVE_PATH, "siam_%s_model_%s_%s_rep_%s" % (dataset, backbone, aug, rep)
    )

    log_file = setup_logging(args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"Using CUDA with {num_gpus} GPU(s)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        num_gpus = 0
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        num_gpus = 0
        print("Using CPU")

    if args.dataset == "dsprites":
        aug, aug_adv = dsprites_augmentations(aug, 64, adv=4 / 255)
        dataset = defaults.get_data(
            args,
            ContrastiveDislibDataset,
            aug=aug,
            aug_adv=aug_adv,
            diet_class=None,
        )
    elif args.dataset == "smallnorb":
        aug, aug_adv = shapes3d_augmentations(aug, 64, adv=4 / 255)
        dataset = defaults.get_data(
            args, ContrastiveDataset, aug=aug, aug_adv=aug_adv, diet_class=None
        )
    elif args.dataset == "shapes3d":
        aug, aug_adv = shapes3d_augmentations(aug, 64, adv=8 / 255)
        dataset = defaults.get_data(
            args, RGBContrastiveDataset, aug=aug, aug_adv=aug_adv, diet_class=None
        )
    else:
        aug, aug_adv = shapes3d_augmentations(aug, 64, adv=8 / 255)
        dataset = defaults.get_data(
            args, RGBContrastiveDataset, aug=aug, aug_adv=aug_adv, diet_class=None
        )
    if backbone != "image":
        train(args, dataset, device, log_file)
