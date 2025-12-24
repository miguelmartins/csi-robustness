# 0: add extra argument on inference for unsup/sup
# 0: INFERENCE SHOULD HAVE NO AUGMENTATIONS
# 1: Implement color augs with gaussian blur
# 2: Implement Diet strength augs
# 3. See how they interact with random noise attack
# 4. See if we can have bigger batch size for diet
# 5: run all combinations
# 6: mention LeJEPA as a proof that the best downstream risk has to be isotropic Gaussian and that poses id. issues
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
from evaluation.identifiability import evaluate


def inference(args, dataset, device, log_file):
    with open(log_file, "w") as file:
        print("\n\nTraining:", file=file)
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        adv_test_dataloader,
        data,
        out_size,
        nc,
        cat_ind,
    ) = dataset

    net = get_model(args.model, nc, out_size, device, args.seed)
    net.load_state_dict(torch.load(os.path.join(args.log_dir, "model.pth")))
    net.eval()
    probe = nn.Linear(512, out_size).to(device)
    optimizer = torch.optim.Adam(list(probe.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(args.num_epochs):
        probe.train()
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
            y_ = probe(z)
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

        probe.eval()
        val_acc = log_validation(
            dataloader=val_dataloader,
            net=net,
            readout=probe,
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
            torch.save(probe.state_dict(), os.path.join(args.log_dir, "probe.pth"))

        def final_logs():
            with open(log_file, "a") as file:
                print("early stopping\n", file=file)
                test_acc = log_validation(
                    dataloader=test_dataloader,
                    net=net,
                    readout=probe,
                    data=data,
                    cat_ind=cat_ind,
                    log_file=log_file,
                    device=device,
                )
                print(
                    "Probe",
                    epoch,
                    "Loss",
                    np.mean(run_loss),
                    "test_acc",
                    test_acc,
                    file=file,
                )
                adv_test_acc = log_validation(
                    dataloader=adv_test_dataloader,
                    net=net,
                    readout=probe,
                    data=data,
                    cat_ind=cat_ind,
                    log_file=log_file,
                    device=device,
                )
                print(
                    "Probe",
                    epoch,
                    "Loss",
                    np.mean(run_loss),
                    "adv_test_acc",
                    adv_test_acc,
                    file=file,
                )

        if best_val_acc > 0.999:
            # change get_data instead of this
            final_logs()
            return
        final_logs()


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
    parser.add_argument("--pretrain", type=str, choices=["supervised", "diet"])

    rep = parser.parse_args().rep
    backbone = parser.parse_args().backbone
    aug = parser.parse_args().aug
    dataset = parser.parse_args().dataset
    pretrain = parser.parse_args().pretrain

    settings = []
    print("Running setting:", "rep:", rep, "dataset:", dataset, "backbone:", backbone)

    args = Args()
    args.seed = defaults.SEED + rep
    args.dataset = dataset
    args.model = backbone
    args.probe = True  # REQUIRED TO LOG PROBE

    if pretrain == "supervised":
        args.log_dir = os.path.join(
            defaults.SAVE_PATH, "%s_model_%s_%s_rep_%s" % (dataset, backbone, aug, rep)
        )
    else:
        args.log_dir = os.path.join(
            defaults.SAVE_PATH,
            "diet_%s_model_%s_%s_rep_%s" % (dataset, backbone, aug, rep),
        )

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
    dataset = defaults.get_data(args, DislibDataset, aug=v2.Identity(), aug_adv=aug_adv)
    log_file = os.path.join(args.log_dir, "probe.txt")
    if backbone != "image":
        inference(args, dataset, device, log_file)
    evaluate(args, dataset, device, os.path.join(args.log_dir, "identifiability.txt"))
