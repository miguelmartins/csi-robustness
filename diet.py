import numpy as np
from scipy.stats import pearsonr as corr

import torch
from torch import nn
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import transforms

import sys
import os
from tqdm import tqdm
import argparse
import shutil

# This builds on https://github.com/facebookresearch/disentangling-correlated-factors
sys.path.append("/home/miguelmartins/Projects/disentangling-correlated-factors")
from datasets.utils import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


SEED = 42
DATA_DIR = "./data"
SAVE_PATH = "./results"


DATASETS = ["dsprites", "shapes3d", "mpi3d", "mpi3d_real", "cars3d", "smallnorb"]


class AddLinfNoise:
    """
    Add uniform L∞-bounded noise to an image tensor (C,H,W) in [0,1].
    """

    def __init__(
        self,
        eps=16 / 255,  # 8 / 255
        p=1.0,
        clip=(0.0, 1.0),
        same_for_all_channels=False,  # False
        generator=None,
    ):
        """
        eps: max per-pixel magnitude (in [0,1] scale). e.g., 8/255 for CIFAR-10.
        p: probability to apply the noise (use <1.0 to sometimes skip).
        clip: min/max clamp after adding noise.
        same_for_all_channels: if True, one noise map shared across channels.
        generator: optional torch.Generator for reproducibility.
        """
        self.eps = float(eps)
        self.p = float(p)
        self.clip = clip
        self.same_for_all_channels = same_for_all_channels
        self.generator = generator

    @torch.no_grad()  # remove if you want gradient through the noise sampling
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be float tensor in [0,1], shape (C,H,W)
        if self.p < 1.0 and torch.rand((), generator=self.generator) > self.p:
            return x
        if self.same_for_all_channels:
            noise = torch.empty(1, x.shape[-2], x.shape[-1], device=x.device).uniform_(
                -self.eps, self.eps, generator=self.generator
            )
            noise = noise.expand_as(x)
        else:
            noise = torch.empty_like(x).uniform_(
                -self.eps, self.eps, generator=self.generator
            )
        x_noisy = x + noise
        return x_noisy.clamp(*self.clip)


def get_transformations_train(aug):
    def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
        # s is the strength of color distortion
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort

    def stronger_distortion():
        # Check Fig. 10 of https://arxiv.org/pdf/2203.13457
        color_jitter = transforms.ColorJitter(1.0, 1.0, 1.0, 0.5)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort

    normalize = transforms.Normalize(mean=0.0, std=1.0)
    base_transform = transforms.Compose([])
    base_transform.transforms.append(normalize)
    if aug == "none":
        pre_transform = base_transform
    elif aug == "crop":
        pre_transform = transforms.Compose(
            [
                transforms.RandomCrop(32),
                transforms.Normalize(mean=0.0, std=1.0),
            ]
        )
    elif aug == "sup":
        pre_transform = transforms.Compose(
            [
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=0.0, std=1.0),
            ]
        )
    elif aug == "simclr":
        pre_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                get_color_distortion(s=0.5),
                transforms.Normalize(mean=0.0, std=1.0),
            ]
        )
    elif aug == "simclr2":
        pre_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                get_color_distortion(s=1),
                transforms.Normalize(mean=0.0, std=1.0),
            ]
        )
    elif aug == "simclr3":
        pre_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                stronger_distortion(),
                transforms.Normalize(mean=0.0, std=1.0),
            ]
        )
    adversarial_transform = transforms.Compose([AddLinfNoise(), normalize])
    return pre_transform, base_transform, adversarial_transform


def get_data(args):
    print("Loading:", args.dataset)
    if args.dataset == "dsprites":
        data = get_dataset(args.dataset)(
            subset=1,
            root=os.path.join(DATA_DIR, args.dataset),
            factors_to_use=["shape", "scale", "orientation", "posX", "posY"],
        )
    else:
        data = get_dataset(args.dataset)(
            subset=1, root=os.path.join(DATA_DIR, args.dataset)
        )

    nc = num_channels[args.dataset]
    cat = categorical[args.dataset][0]
    cat_ind = [j for j, item in enumerate(data.lat_names) if item == cat][0]
    if args.debug:
        print("Target", cat, cat_ind)

    # split train, val, test
    test_ratio = 0.3
    N = data.imgs.shape[0]
    if args.debug:
        print("Total Data", N)
    num_test = int(N * test_ratio)
    num_train = N - num_test

    np.random.seed(args.seed)
    indices = np.random.choice(N, N, replace=False)
    test_ind = indices[num_train:]
    train_ind = indices[:num_train]
    if args.debug:
        print(
            "num_test",
            len(test_ind),
            "num_train",
            len(train_ind),
            "sum",
            len(test_ind) + len(train_ind),
        )

    images = data.imgs.copy()
    if nc == 1:
        images = images.reshape(-1, 1, 64, 64)
    else:
        if images.shape[-1] == 3:
            images = np.transpose(images, (0, 3, 1, 2))
    # images = images / 255.0
    if args.debug:
        print("images", images.shape, images.dtype)
    targets = data.lat_values.copy()
    for i in range(targets.shape[1]):
        tmp = np.unique(targets[:, i])
        targets[:, i] /= tmp[1]
    targets = targets.astype(int)

    if args.debug:
        for i in range(targets.shape[1]):
            if i == cat_ind:
                print(f"cat_ind, {i}")
            print(i, np.unique(targets[:, i]))
        print("targets", targets.shape, targets.dtype)
        print(targets)
    train, base, adversarial = get_transformations_train(aug=args.aug)

    train_data = DislibDataset(
        images[train_ind],
        targets[train_ind],
        transform=train,
    )
    train_data_diet = DietDataset(
        train_data,
        transform=base,
    )
    test_data = DislibDataset(
        images[test_ind],
        targets[test_ind],
        transform=base,
    )
    adv_test_data = DislibDataset(
        images[test_ind],
        targets[test_ind],
        transform=adversarial,
    )

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    train_diet_dataloader = DataLoader(
        train_data_diet, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    adv_test_dataloader = DataLoader(
        adv_test_data, batch_size=args.batch_size, shuffle=False
    )

    if args.dataset == "cars3d":
        # missing classes in cars3d
        out_size = 183
    else:
        out_size = len(np.unique(targets[:, cat_ind]))

    return (
        train_dataloader,
        # train_diet_dataloader,
        test_dataloader,
        adv_test_dataloader,
        data,
        out_size,
        nc,
        cat_ind,
    )


# sort latents
categorical = {
    "dsprites": ["shape"],
    "shapes3d": ["objType"],
    "mpi3d": ["objShape"],
    "mpi3d_real": ["objShape"],
    "cars3d": ["object_type"],
    "smallnorb": ["category", "instance"],
}
continuous = {
    "dsprites": ["scale", "posX", "posY"],
    "shapes3d": ["objSize", "objAzimuth"],
    "mpi3d": ["posX", "posY"],
    "mpi3d_real": ["posX", "posY"],
    "cars3d": ["elevation"],
    "smallnorb": ["elevation"],
}
manifold = {
    "dsprites": ["orientation"],
    "shapes3d": ["floorCol", "wallCol", "objCol"],
    "mpi3d": [],
    "mpi3d_real": [],
    "cars3d": ["azimuth"],
    "smallnorb": ["rotation"],
}
other = {
    "dsprites": [],
    "shapes3d": [],
    "mpi3d": ["objCol", "objSize", "cameraHeight", "backCol"],
    "mpi3d_real": ["objCol", "objSize", "cameraHeight", "backCol"],
    "cars3d": [],
    "smallnorb": ["lighting"],
}

num_channels = {
    "dsprites": 1,
    "shapes3d": 3,
    "mpi3d": 3,
    "mpi3d_real": 3,
    "cars3d": 3,
    "smallnorb": 1,
}


class DietDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, idx

    def __len__(self):
        return len(self.dataset)


class DislibDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx]).float()
        y = torch.tensor(self.labels[idx])

        if self.transform is not None:
            x = self.transform(x)
        return x, y


def get_model(model, nc, out_size, seed):
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


def log_val(data, net, readout, cat_ind, epoch, run_loss, dataloader, name="val"):
    Y, Z, Y_ = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
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
    acc_ = np.mean(Y[:, cat_ind] == Y_)
    print(f"{name}_acc", acc_)
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
            name,
            " Epoch",
            epoch,
            "Loss",
            np.mean(run_loss),
            f"{name}_acc",
            acc_,
            "\n",
            file=file,
        )
    return acc_


def train(args, dataset, log_file):
    with open(log_file, "a") as file:
        print("\n\nTraining:", file=file)
    (
        train_dataloader,
        test_dataloader,
        adv_test_dataloader,
        data,
        out_size,
        nc,
        cat_ind,
    ) = dataset
    net = get_model(args.model, nc, out_size, args.seed)
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
                print("y_cat", y[:, cat_ind])
            loss = criterion(y_, y[:, cat_ind].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(run_loss))

        net.eval()
        val_acc = log_val(
            data, net, readout, cat_ind, epoch, run_loss, test_dataloader, name="val"
        )
        log_val(
            data,
            net,
            readout,
            cat_ind,
            epoch,
            run_loss,
            adv_test_dataloader,
            name="adv",
        )

        net.eval()
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
        # torch.save(net.state_dict(), os.path.join(args.log_dir, "model.pth"))
        # torch.save(readout.state_dict(), os.path.join(args.log_dir, "readout.pth"))


def evaluate(args, dataset, log_file):
    with open(log_file, "a") as file:
        print("\n\nEvaluating:", file=file)
    (
        train_dataloader,
        test_dataloader,
        adv_test_dataloader,
        data,
        out_size,
        nc,
        cat_ind,
    ) = dataset
    net = get_model(args.model, nc, out_size, args.seed)

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
        for i, (x, y) in enumerate(test_dataloader):
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


def setup_logging(args):
    # Ensure the log directory exists and is empty
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)  # Remove the directory and all its contents
    os.makedirs(args.log_dir)  # Recreate the directory
    log_file = os.path.join(args.log_dir, "log.txt")
    with open(log_file, "a") as file:
        print(args, file=file)
    return log_file


class Args:
    def __init__(self):
        self.dataset = "dsprites"
        self.seed = SEED
        self.batch_size = int(2**12)
        self.model = "cnn"
        self.lr = 1e-3
        self.num_epochs = 20
        self.log_dir = SAVE_PATH
        self.debug = False
        self.aug = "none"

    def __str__(self):
        return "Args:\n" + "\n".join(
            f"{attr}: {value}"
            for attr, value in self.__dict__.items()
            if not attr.startswith("__")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script LinRep")
    parser.add_argument(
        "--setting", type=int, default=0, help="train setting see below"
    )
    parser.add_argument("--aug", type=str, default="none", help="train augs")
    setting = parser.parse_args().setting
    aug = parser.parse_args().aug

    settings = []
    for rep in range(10):
        for dataset in DATASETS[:]:
            for model in ["cnn"]:
                settings.append([rep, dataset, model])
    assert setting >= 0 and setting < len(settings)
    rep, dataset, model = settings[setting]
    print("Running setting:", "rep:", rep, "dataset:", dataset, "model:", model)

    args = Args()
    args.seed = SEED + rep
    args.dataset = dataset
    args.model = model
    args.aug = aug
    args.log_dir = os.path.join(
        SAVE_PATH, "diet_aug_%s_%s_model_%s_rep_%s" % (aug, dataset, model, rep)
    )

    args.num_epochs = 100
    log_file = setup_logging(args)
    dataset = get_data(args)
    if model != "image":
        train(args, dataset, log_file)
    evaluate(args, dataset, log_file)
