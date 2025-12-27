# sort latents
import numpy as np
import os
import torch
import sys

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
SEED = 42
DATA_DIR = "./data2"
SAVE_PATH = "./logs"
DATASETS = ["dsprites", "shapes3d", "mpi3d", "mpi3d_real", "cars3d", "smallnorb"]

sys.path.append("/home/miguelmartins/Projects/disentangling-correlated-factors")
from datasets.utils import get_dataset


def get_data(args, dataset_class, aug, aug_adv, diet_class=None):
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
    print("Target", cat, cat_ind)

    # split train, val, test
    val_ratio = 0.1
    test_ratio = 0.1
    N = data.imgs.shape[0]
    print("Total Data", N)
    num_val = int(N * val_ratio)
    num_test = int(N * test_ratio)
    num_train = N - num_val - num_test

    np.random.seed(args.seed)
    indices = np.random.choice(N, N, replace=False)
    test_ind = indices[: num_val + num_test]
    train_ind = indices[num_val + num_test :]

    print(
        "num_test",
        len(test_ind),
        "num_train",
        len(train_ind),
        "sum",
        len(test_ind) + len(test_ind) + len(train_ind),
    )

    # legacy, encode variables in numerical
    images = data.imgs.copy()
    if args.debug:
        print("images", images.shape, images.dtype)

    targets = data.lat_values.copy()
    for i in range(targets.shape[1]):
        tmp = np.unique(targets[:, i])
        targets[:, i] /= tmp[1]
    targets = targets.astype(int)
    ##### end legacy #####

    for i in range(targets.shape[1]):
        print(i, np.unique(targets[:, i]))
    print("targets", targets.shape, targets.dtype)

    if diet_class is not None:
        train_data = diet_class(
            images[train_ind], torch.tensor(targets[train_ind]), augmentations=aug
        )
    else:
        train_data = dataset_class(
            images[train_ind], torch.tensor(targets[train_ind]), augmentations=aug
        )

    test_data = dataset_class(
        images[test_ind],
        torch.tensor(targets[test_ind]),
    )
    adv_test_data = dataset_class(
        images[test_ind], torch.tensor(targets[test_ind]), augmentations=aug_adv
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        multiprocessing_context="fork",
        prefetch_factor=4,
        persistent_workers=True,
        num_workers=8,
        drop_last=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        multiprocessing_context="fork",
        prefetch_factor=4,
        persistent_workers=True,
        num_workers=8,
        drop_last=False,
    )
    adv_test_dataloader = torch.utils.data.DataLoader(
        adv_test_data,
        batch_size=args.batch_size,
        shuffle=False,
        multiprocessing_context="fork",
        prefetch_factor=4,
        persistent_workers=True,
        num_workers=8,
        drop_last=False,
    )
    # adv_test_dataloader = DataLoader(
    #     adv_test_data,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     multiprocessing_context="fork",
    #     prefetch_factor=4,
    #     persistent_workers=True,
    #     num_workers=8,
    #     drop_last=False,
    # )

    if args.dataset == "cars3d":
        # missing classes in cars3d
        out_size = 183
    else:
        out_size = len(np.unique(targets[:, cat_ind]))

    return (
        train_dataloader,
        test_dataloader,
        adv_test_dataloader,
        data,
        out_size,
        nc,
        cat_ind,
    )
