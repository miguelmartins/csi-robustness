import argparse
import dislib.defaults as defaults
import numpy as np
import os
import torch
import torch.nn as nn
import shutil

from dataset_processing.augmentations import dsprites_augmentations
from dataset_processing.load_datasets import DislibDataset
from torchvision.models import resnet18
from tqdm.auto import tqdm
from scipy.stats import pearsonr as corr


class Args:
    def __init__(self):
        self.dataset = "dsprites"
        self.seed = defaults.SEED
        self.batch_size = int(2**12)
        self.model = "cnn"
        self.lr = 1e-3
        self.num_epochs = 100
        self.log_dir = defaults.SAVE_PATH
        self.debug = False

    def __str__(self):
        return "Args:\n" + "\n".join(
            f"{attr}: {value}"
            for attr, value in self.__dict__.items()
            if not attr.startswith("__")
        )


def setup_logging(args):
    # Ensure the log directory exists and is empty
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)  # Remove the directory and all its contents
    os.makedirs(args.log_dir)  # Recreate the directory
    log_file = os.path.join(args.log_dir, "log.txt")
    with open(log_file, "a") as file:
        print(args, file=file)
    return log_file


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


def LinReg(X, Y):
    beta = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return X @ beta


def train(args, dataset, device, log_file):
    with open(log_file, "a") as file:
        print("\n\nTraining:", file=file)
    (train_dataloader, val_dataloader, test_dataloader, data, out_size, nc, cat_ind) = (
        dataset
    )

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
            x = x.to(torch.float32).to(device)
            y = y.to(torch.long)
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
        Y, Z, Y_ = [], [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(val_dataloader):
                x = x.to(torch.float32).to(device)
                z = net(x)
                y_ = readout(z)
                Y.append(y.to(torch.long).detach().cpu().numpy())
                Y_.append(y_.argmax(1).detach().cpu().numpy())
                Z.append(z.detach().cpu().numpy())
                if args.debug:
                    break
        Y = np.concatenate(Y)
        Y_ = np.concatenate(Y_)
        Z = np.concatenate(Z)
        val_acc = np.mean(Y[:, cat_ind] == Y_)
        print("val_acc", val_acc)
        Y_ = LinReg(Z, Y)
        with open(log_file, "a") as file:
            for i in range(Y.shape[1]):
                print(
                    "Coordinate",
                    i,
                    data.lat_names[i],
                    corr(Y[:, i], Y_[:, i])[0],
                    file=file,
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


def evaluate(args, dataset, device, log_file):
    with open(log_file, "a") as file:
        print("\n\nEvaluating:", file=file)
    (train_dataloader, val_dataloader, test_dataloader, data, out_size, nc, cat_ind) = (
        dataset
    )
    net = get_model(args.model, nc, out_size, device, args.seed)

    # prepare model
    if args.model == "image":
        pass
    else:
        net.load_state_dict(torch.load(os.path.join(args.log_dir, "model.pth")))
        net.eval()

    x_train, y_train, x_val, y_val = [], [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(torch.float32).to(device)
            y_train.append(y.to(torch.long).detach().cpu().numpy())
            x_train.append(net(x).detach().cpu().numpy())
            if args.debug:
                break
        for i, (x, y) in enumerate(val_dataloader):
            x = x.to(torch.float32).to(device)
            y_val.append(y.to(torch.long).detach().cpu().numpy())
            x_val.append(net(x).detach().cpu().numpy())
            if args.debug:
                break
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    if args.debug:
        with open(log_file, "a") as file:
            print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, file=file)

    # decode all coordinates
    tmp = np.linalg.pinv(x_train.T @ x_train) @ x_train.T
    for i in range(y_train.shape[1]):
        y = y_train[:, i].copy() * 1.0
        y -= np.mean(y)
        y /= np.std(y)
        beta = tmp @ y
        y_train_ = x_train @ beta
        y_val_ = x_val @ beta
        with open(log_file, "a") as file:
            print(
                "Coordinate",
                i,
                data.lat_names[i],
                "\ntrain",
                corr(y_train[:, i], y_train_),
                "\nval",
                corr(y_val[:, i], y_val_),
                file=file,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script LinRep")
    parser.add_argument("--rep", type=int, default=0, help="repetetion")
    rep = parser.parse_args().rep

    parser.add_argument(
        "--backbone", type=str, default="cnn", help="mlp or cnn (resnet18)"
    )

    backbone = parser.parse_args().backbone

    parser.add_argument(
        "--aug", type=str, default="none", help="Augmentations in train"
    )
    aug = parser.parse_args().aug

    parser.add_argument("--dataset", type=str, default="dsprites", help="Dataset")
    dataset = parser.parse_args().dataset

    settings = []
    print("Running setting:", "rep:", rep, "dataset:", dataset, "backbone:", backbone)

    args = Args()
    args.seed = defaults.SEED + rep
    args.dataset = dataset
    args.model = backbone
    args.log_dir = os.path.join(
        defaults.SAVE_PATH, "%s_model_%s_rep_%s" % (dataset, backbone, rep)
    )

    log_file = setup_logging(args)
    dataset = defaults.get_data(args, DislibDataset, aug=aug)
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
    if backbone != "image":
        train(args, dataset, device, log_file)
    evaluate(args, dataset, device, log_file)
