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

from dataset_processing.augmentations import dsprites_augmentations
from dataset_processing.load_datasets import DislibDataset
from tqdm.auto import tqdm

from evaluation.identifiability import log_test_evaluation, log_validation
from evaluation.logging import Args, setup_logging
from models.baselines import get_model
from torchvision.transforms import v2

from optimization.scheduler import build_optimizer_and_scheduler
from dataset_processing.load_datasets import DietDataset
from evaluation.identifiability import evaluate


def train(args, dataset, device, log_file):
    with open(log_file, "a") as file:
        print("\n\nTraining:", file=file)
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        _,
        data,
        out_size,
        nc,
        cat_ind,
    ) = dataset

    net = get_model(args.model, nc, out_size, device, args.seed)
    readout = nn.Linear(512, len(train_dataloader.dataset)).to(device)
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     num_gpus = torch.cuda.device_count()
    # else:
    #     num_gpus = 0
    #
    # # after creating net/readout:
    # if num_gpus > 1:
    #     net = nn.DataParallel(net)
    #     readout = nn.DataParallel(readout)
    optimizer, scheduler = build_optimizer_and_scheduler(
        [net, readout], args.num_epochs
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(args.num_epochs):
        net.train()
        run_loss = []
        progress_bar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
        )
        for x, y in progress_bar:
            x = x.to(device)  # .to(torch.float32).to(device)
            y = y.to(device)  # .to(torch.long)
            z = net(x)
            y_ = readout(z)
            loss = criterion(y_, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(run_loss))
            if args.debug:
                break

        scheduler.step()
    torch.save(net.state_dict(), os.path.join(args.log_dir, "model.pth"))
    torch.save(readout.state_dict(), os.path.join(args.log_dir, "readout.pth"))


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
    args.log_dir = os.path.join(
        defaults.SAVE_PATH, "diet_%s_model_%s_%s_rep_%s" % (dataset, backbone, aug, rep)
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

    aug, aug_adv = dsprites_augmentations(aug, 64, adv=4 / 255)
    dataset = defaults.get_data(
        args, DislibDataset, aug=aug, aug_adv=aug_adv, diet_class=DietDataset
    )
    if backbone != "image":
        train(args, dataset, device, log_file)
    del dataset
    dataset = defaults.get_data(
        args, DislibDataset, aug=aug, aug_adv=aug_adv, diet_class=None
    )
    evaluate(args, dataset, device, os.path.join(args.log_dir, "identifiability.txt"))
