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

from evaluation.identifiability import evaluate, log_test_evaluation, log_validation
from evaluation.logging import Args, setup_logging
from models.baselines import get_model
from torchvision.transforms import v2
from scipy.stats import pearsonr as corr


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

    aug, aug_adv = dsprites_augmentations(aug, 64, adv=0.01)
    dataset = defaults.get_data(args, DislibDataset, aug=v2.Identity(), aug_adv=aug_adv)
    log_file = os.path.join(args.log_dir, "identifiability.txt")
    evaluate(args, dataset, device, log_file)
