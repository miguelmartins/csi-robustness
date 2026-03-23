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
from dataset_processing.load_datasets import (
    BeforeAttack,
    DislibDataset,
    MPI3DDataset,
    RGBBeforeAttack,
    RGBDataset,
)
from tqdm.auto import tqdm

from evaluation.adversarial import evaluate_adversarial, evaluate_adversarial_hf
from evaluation.identifiability import evaluate, log_test_evaluation, log_validation
from evaluation.logging import Args, setup_logging
from models.baselines import get_model
from torchvision.transforms import v2
from scipy.stats import pearsonr as corr
from dataset_processing.augmentations import shapes3d_augmentations
from dataset_processing.load_datasets import GrayDataset


class ResizeBeforeAttack(BeforeAttack):
    def __init__(
        self,
        images,
        labels,
        normalize=None,
        resize=v2.Resize(
            size=(224, 224),  # Replace with your target benchmark size
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        augmentations=None,
    ):
        # Call the parent constructor and explicitly set nch=3
        super().__init__(
            images=images,
            labels=labels,
            normalize=normalize,
            resize=resize,
            augmentations=augmentations,
        )


class ResizeRGBBeforeAttack(BeforeAttack):
    def __init__(
        self,
        images,
        labels,
        normalize=None,
        resize=v2.Resize(
            size=(224, 224),  # Replace with your target benchmark size
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        augmentations=None,
    ):
        # Call the parent constructor and explicitly set nch=3
        super().__init__(
            images=images,
            labels=labels,
            normalize=normalize,
            resize=resize,
            augmentations=augmentations,
        )


fm_dict = {"v3conv": "facebook/dinov3-convnext-base-pretrain-lvd1689m"}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script LinRep")
    parser.add_argument("--rep", type=int, default=0, help="repetetion")
    parser.add_argument(
        "--backbone", type=str, default="cnn", help="mlp or cnn (resnet18)"
    )
    parser.add_argument("--dataset", type=str, default="dsprites", help="Dataset")
    parser.add_argument("--pretrain", type=str, choices=["supervised", "diet"])

    parser.add_argument(
        "--fm",
        type=str,
        default="v3conv",
        choices=["v3conv"],
        help="Foundational model",
    )
    rep = parser.parse_args().rep
    backbone = parser.parse_args().backbone
    dataset = parser.parse_args().dataset
    pretrain = parser.parse_args().pretrain
    fm = parser.parse_args().fm
    fm_config = fm_dict[fm]
    settings = []
    print("Running setting:", "rep:", rep, "dataset:", dataset, "backbone:", backbone)

    args = Args()
    args.seed = defaults.SEED + rep
    args.dataset = dataset
    args.model = backbone
    args.probe = True  # REQUIRED TO LOG PROBE

    if pretrain == "supervised":
        args.log_dir = os.path.join(
            defaults.SAVE_PATH, "%s_model_%s_rep_%s" % (dataset, backbone, rep)
        )
    else:
        args.log_dir = os.path.join(
            defaults.SAVE_PATH,
            "%s_%s_model_%s_rep_%s" % (fm, dataset, backbone, rep),
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

    log_file = os.path.join(args.log_dir, "pgd.txt")
    aug = "none"
    if args.dataset == "dsprites":
        adv = 4 / 255
        aug, aug_adv = shapes3d_augmentations(aug, 64, adv=adv)
        dataset = defaults.get_data(
            args, ResizeBeforeAttack, aug=aug, aug_adv=v2.Identity(), diet_class=None
        )
        evaluate_adversarial_hf(
            args, dataset, device, log_file, eps=adv, fm=fm_config
        )  # TODO: if results weird build BeforeAttack scale false
    elif args.dataset == "smallnorb":
        adv = 4 / 255
        aug, aug_adv = shapes3d_augmentations(aug, 64, adv=adv)
        dataset = defaults.get_data(
            args, ResizeBeforeAttack, aug=aug, aug_adv=v2.Identity(), diet_class=None
        )
        evaluate_adversarial_hf(args, dataset, device, log_file, eps=adv, fm=fm_config)
    elif args.dataset == "shapes3d":
        adv = 8 / 255
        aug, aug_adv = shapes3d_augmentations(aug, 64, adv=adv)
        dataset = defaults.get_data(
            args, ResizeRGBBeforeAttack, aug=aug, aug_adv=v2.Identity(), diet_class=None
        )
        evaluate_adversarial_hf(
            args, dataset, device, log_file, eps=adv, nch=3, fm=fm_config
        )
    elif args.dataset == "cars3d":
        adv = 8 / 255
        aug, aug_adv = shapes3d_augmentations(aug, 64, adv=adv)
        dataset = defaults.get_data(
            args, ResizeBeforeAttack, aug=aug, aug_adv=v2.Identity(), diet_class=None
        )
        evaluate_adversarial_hf(
            args, dataset, device, log_file, eps=adv, nch=3, fm=fm_config
        )
    else:
        adv = 8 / 255
        aug, aug_adv = shapes3d_augmentations(aug, 64, adv=adv)
        dataset = defaults.get_data(
            args, ResizeRGBBeforeAttack, aug=aug, aug_adv=v2.Identity(), diet_class=None
        )  # TODO: If results are weird try building new class for MPI3DDataset
        evaluate_adversarial_hf(
            args, dataset, device, log_file, eps=adv, nch=3, fm=fm_config
        )
