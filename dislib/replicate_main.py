import argparse
import dislib.defaults as defaults
import os

from dataset_processing.augmentations import dsprites_augmentations
from dataset_processing.load_datasets import DislibDataset
from dislib.main import setup_logging


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


def main():
    pass


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
    main()
