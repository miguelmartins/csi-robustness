import gc
import numpy as np
import torch
import os
import pathlib
from scipy.stats import pearsonr as corr
from models.baselines import get_model
import torch
import torch.nn as nn
from torchvision.transforms import v2
from tqdm.auto import tqdm
from models.baselines import SimSiam
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


def pgd_attack(
    model, images, labels, eps=8 / 255, alpha=2 / 255, iters=40, nch=1, fm=False
):
    """
    Executes a Projected Gradient Descent (PGD) attack on a batch of images.

    Args:
        model: The PyTorch model to attack.
        images: Batch of original images (Tensor).
        labels: Batch of true labels (Tensor).
        eps: Maximum perturbation magnitude.
        alpha: Step size per iteration.
        iters: Number of iterations.
    """
    # Detach to ensure we don't accidentally modify the original dataset tensors
    images = images.clone().detach()
    labels = labels.clone().detach()

    # Initialize adversarial examples
    # (Optional but recommended: add random initialization within the eps ball)
    x_adv = images.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    loss_fn = nn.CrossEntropyLoss()

    for i in range(iters):
        # Enable gradient tracking for the input image
        x_adv.requires_grad = True

        # 1. Forward pass

        outputs = model(x_adv)
        if fm is True:
            outputs = outputs.pooler_output
        model.zero_grad()
        loss = loss_fn(outputs, labels)

        # 2. Backward pass to get the gradient of the loss w.r.t the image
        loss.backward()

        with torch.no_grad():
            # 3. Take a step in the direction of the gradient sign
            adv_images = x_adv + alpha * x_adv.grad.sign()

            # 4. Project the perturbation back into the L_inf epsilon ball
            eta = torch.clamp(adv_images - images, min=-eps, max=eps)

            # 5. Apply the clipped perturbation and ensure pixel values are valid [0, 1]
            x_adv = torch.clamp(images + eta, min=0, max=1).detach()

    return x_adv


def pgd_attack_norm(
    model,
    images,
    labels,
    eps=8 / 255,
    alpha=2 / 255,
    iters=40,
    mean=None,
    std=None,
    fm=False,
):
    images = images.clone().detach()
    labels = labels.clone().detach()

    # Initialize adversarial examples with random noise
    x_adv = images.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    loss_fn = nn.CrossEntropyLoss()

    for i in range(iters):
        x_adv.requires_grad = True

        # --- THE FIX: Normalize JUST for the model forward pass ---
        # If mean/std are provided, apply them here
        if mean is not None and std is not None:
            # Reshape mean/std for broadcasting [1, C, 1, 1]
            x_input = v2.Normalize(mean=mean, std=std)(x_adv)
        else:
            x_input = x_adv

        # 1. Forward pass with normalized input
        outputs = model(x_input)

        if fm:
            outputs = outputs.pooler_output

        model.zero_grad()
        loss = loss_fn(outputs, labels)

        # 2. Backward pass
        # Gradients are backpropagated through normalization to the raw x_adv
        loss.backward()

        with torch.no_grad():
            # 3. Update step (performed on unnormalized x_adv)
            adv_images = x_adv + alpha * x_adv.grad.sign()

            # 4. Projection
            eta = torch.clamp(adv_images - images, min=-eps, max=eps)

            # 5. Clamp to valid pixel range [0, 1]
            x_adv = torch.clamp(images + eta, min=0, max=1).detach()

    return x_adv


def evaluate_adversarial(
    args,
    dataset,
    device,
    log_file,
    iteration=0,
    eps=8 / 255,
    alpha=2 / 255,
    iters=40,
    nch=1,
    framework="diet",
):
    if iteration == 0:
        file_mode = "w"
    else:
        file_mode = "a"
    with open(log_file, "w") as file:
        print(f"\n\nEvaluating {iteration}:", file=file)
    (
        train_dataloader,
        test_dataloader,
        adv_test_dataloader,
        data,
        out_size,
        nc,
        cat_ind,
    ) = dataset
    net = get_model(args.model, nc, out_size, device, args.seed)
    checkpoint_path = os.path.join(args.log_dir, "model.pth")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if framework == "siam":
        print("Loading Siam...")
        siam_model = SimSiam(backbone=net)
        siam_model.projector.set_layers(2)
        siam_model.load_state_dict(state_dict)
        net = siam_model.encoder.to(device)
    else:
        net.load_state_dict(state_dict)

    net.eval()
    x_train, y_train, x_val, y_val = [], [], [], []
    x_adv, y_adv = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(torch.float32).to(device)
            x = v2.Normalize(mean=[0.5] * nch, std=[0.5] * nch)(x)
            y_train.append(y.to(torch.long).detach().cpu().numpy())
            x_train.append(net(x).detach().cpu().numpy())
            if args.debug:
                break
        for i, (x, y) in enumerate(test_dataloader):
            x = x.to(torch.float32).to(device)
            x = v2.Normalize(mean=[0.5] * nch, std=[0.5] * nch)(x)
            y_val.append(y.to(torch.long).detach().cpu().numpy())
            x_val.append(net(x).detach().cpu().numpy())
    for i, (x, y) in tqdm(enumerate(adv_test_dataloader)):
        x = x.to(torch.float32).to(device)
        x = pgd_attack(
            model=net,
            images=x,
            labels=y[:, cat_ind].to(torch.long).to(device),
            eps=eps,
            alpha=alpha,
            iters=iters,
        )
        x = v2.Normalize(mean=[0.5] * nch, std=[0.5] * nch)(x)
        y_adv.append(y.to(torch.long).detach().cpu().numpy())
        x_adv.append(net(x).detach().cpu().numpy())
        if args.debug:
            break
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    x_adv = np.concatenate(x_adv)
    y_adv = np.concatenate(y_adv)
    if args.debug:
        with open(log_file, "a") as file:
            print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, file=file)

    print("Probing onto ", log_file)
    # decode all coordinates
    tmp = np.linalg.pinv(x_train.T @ x_train) @ x_train.T
    for i in range(y_train.shape[1]):
        y = y_train[:, i].copy() * 1.0
        y -= np.mean(y)
        y /= np.std(y)
        beta = tmp @ y
        y_train_ = x_train @ beta
        y_val_ = x_val @ beta
        y_adv_ = x_adv @ beta
        with open(log_file, "a") as file:
            print(
                "Coordinate",
                i,
                data.lat_names[i],
                "\ntrain",
                corr(y_train[:, i], y_train_),
                "\nval",
                corr(y_val[:, i], y_val_),
                "\nadv",
                corr(y_adv[:, i], y_adv_),
                file=file,
            )


def compute_embeddings_fm(
    args,
    dataset,
    device,
    iteration=0,
    nch=1,
    fm=None,
):
    # 1. Create the parent directory if it doesn't exist
    # parents=True creates nested folders; exist_ok=True prevents errors if it's already there
    embeddings_dir = pathlib.Path(args.log_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    (
        train_dataloader,
        test_dataloader,
        adv_test_dataloader,
        data,
        out_size,
        nc,
        cat_ind,
    ) = dataset
    processor = AutoImageProcessor.from_pretrained(fm)
    net = AutoModel.from_pretrained(
        fm,
        device_map="auto",
    )
    net.eval()

    x_train, y_train, x_val, y_val = [], [], [], []
    x_adv, y_adv = [], []
    with torch.no_grad():
        for i, (x, y) in tqdm(
            enumerate(train_dataloader), desc="Extracting Train Features"
        ):
            x = processor(images=x, return_tensors="pt").to(net.device)
            y_train.append(y.to(torch.long).detach().cpu().numpy())
            x_train.append(net(**x).pooler_output.detach().cpu().numpy())
        for i, (x, y) in tqdm(
            enumerate(test_dataloader), desc="Extracting Val Features"
        ):
            x = processor(images=x, return_tensors="pt").to(net.device)
            y_val.append(y.to(torch.long).detach().cpu().numpy())
            x_val.append(net(**x).pooler_output.detach().cpu().numpy())

    _mean = processor.image_mean
    _std = processor.image_std
    final_data = {
        "x_train": np.concatenate(x_train, axis=0),
        "y_train": np.concatenate(y_train, axis=0),
        "x_val": np.concatenate(x_val, axis=0),
        "y_val": np.concatenate(y_val, axis=0),
    }
    embeddings_path = embeddings_dir / "embeddings.npz"

    # 2. Physically create the directory
    # parents=True: Creates any missing folders in the path (e.g., ./results/run1/)
    # exist_ok=True: Prevents an error if the folder already exists
    np.savez_compressed(embeddings_path, **final_data)


def evaluate_adversarial_hf(
    args,
    dataset,
    device,
    log_file,
    iteration=0,
    eps=8 / 255,
    alpha=2 / 255,
    iters=40,
    nch=1,
    fm=None,
):
    # 1. Create the parent directory if it doesn't exist
    # parents=True creates nested folders; exist_ok=True prevents errors if it's already there
    pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    if iteration == 0:
        file_mode = "w"
    else:
        file_mode = "a"
    with open(log_file, "w") as file:
        print(f"\n\nEvaluating {iteration}:", file=file)
    (
        train_dataloader,
        test_dataloader,
        adv_test_dataloader,
        data,
        out_size,
        nc,
        cat_ind,
    ) = dataset
    processor = AutoImageProcessor.from_pretrained(fm)
    net = AutoModel.from_pretrained(
        fm,
        device_map="auto",
    )
    net.eval()
    embeddings = np.load(os.path.join(args.log_dir, "embeddings.npz"))
    x_train = embeddings["x_train"]
    y_train = embeddings["y_train"]
    x_val = embeddings["x_val"]
    y_val = embeddings["y_val"]
    x_adv, y_adv = [], []

    _mean = processor.image_mean
    _std = processor.image_std
    for i, (x, y) in tqdm(enumerate(adv_test_dataloader), desc="Adversarial Attack"):
        x = processor(images=x, return_tensors="pt", do_normalize=False)[
            "pixel_values"
        ].to(net.device)
        x = pgd_attack_norm(
            model=net,
            images=x,
            labels=y[:, cat_ind].to(torch.long).to(net.device),
            eps=eps,
            alpha=alpha,
            iters=iters,
            mean=_mean,
            std=_std,
            fm=True,
        )
        # TODO: Watch out for this!
        x = v2.Normalize(mean=_mean, std=_std)(x)
        y_adv.append(y.to(torch.long).detach().cpu().numpy())
        with torch.no_grad():
            x = net(pixel_values=x).pooler_output.detach().cpu().numpy()
        x_adv.append(x)
    x_adv = np.concatenate(x_adv)
    y_adv = np.concatenate(y_adv)
    if args.debug:
        with open(log_file, "a") as file:
            print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, file=file)
    if "net" in locals():
        net.cpu()
        del net

    # 2. Delete the processor and any other remaining GPU tensors
    if "processor" in locals():
        del processor

    # 3. Explicitly trigger Python garbage collection

    gc.collect()
    print("Probing onto ", log_file)
    # decode all coordinates
    tmp = np.linalg.pinv(x_train.T @ x_train) @ x_train.T
    for i in range(y_train.shape[1]):
        y = y_train[:, i].copy() * 1.0
        y -= np.mean(y)
        y /= np.std(y)
        beta = tmp @ y
        y_train_ = x_train @ beta
        y_val_ = x_val @ beta
        y_adv_ = x_adv @ beta
        with open(log_file, "a") as file:
            print(
                "Coordinate",
                i,
                data.lat_names[i],
                "\ntrain",
                corr(y_train[:, i], y_train_),
                "\nval",
                corr(y_val[:, i], y_val_),
                "\nadv",
                corr(y_adv[:, i], y_adv_),
                file=file,
            )
