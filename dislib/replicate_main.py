# 0: make it so it does not delete the directory
# 1: modify train and evaluate
# 2: modify np.linalg.pinv to torch
# 3: adapt to DIET
# 4: run all combinations
# 5: start working on next dataset
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
    readout = nn.Linear(512, out_size).to(device)

    optimizer = torch.optim.Adam(
        list(net.parameters()) + list(readout.parameters()), lr=args.lr
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
            if args.debug:
                print("y", y.shape, y.dtype, y.min(), y.max(), y)
                print("y_", y_.shape, y_.dtype, y_.min(), y_.max(), y_)
            loss = criterion(y_, y[:, cat_ind].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(run_loss))
            if args.debug:
                break

        net.eval()
        val_acc = log_validation(
            dataloader=val_dataloader,
            net=net,
            readout=readout,
            data=data,
            cat_ind=cat_ind,
            log_file=log_file,
            device=device,
        )
        with open(log_file, "a") as file:
            print(
                "Epoch", epoch, "Loss", np.mean(run_loss), "val_acc", val_acc, file=file
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), os.path.join(args.log_dir, "model.pth"))
            torch.save(readout.state_dict(), os.path.join(args.log_dir, "readout.pth"))
        if args.debug:
            break
        if best_val_acc > 0.999:
            with open(log_file, "a") as file:
                print("early stopping", file=file)
            break


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
    args.log_dir = os.path.join(
        defaults.SAVE_PATH, "%s_model_%s_%s_rep_%s" % (dataset, backbone, aug, rep)
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

    aug, aug_adv = dsprites_augmentations(aug, 64, adv=8 / 255)
    dataset = defaults.get_data(args, DislibDataset, aug=aug, aug_adv=v2.Identity())
    if backbone != "image":
        train(args, dataset, device, log_file)
    log_test_evaluation(args, dataset, device, log_file)
