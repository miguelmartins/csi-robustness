from torchvision.transforms import v2

from dataset_processing.attacks import UniformLinfAttackNoClamp
import torch


def dsprites_symmetries(
    out_size: int = 64,
    max_translate_frac: float = 0.25,
    max_rotate_deg: float = 180.0,
    # scale_range=(0.6, 1.4),  # Not sure about this
    use_reflection: bool = False,
):
    # TODO: warning, this does leave some artifacts, distorting ever so slightly the shapes.
    # One should be careful if this does not confound why this set of augs works so well.
    tfms = [
        v2.RandomAffine(
            degrees=max_rotate_deg,
            translate=(max_translate_frac, max_translate_frac),
            # scale=scale_range,
            shear=None,  # critical: no shear (would change shape)
            interpolation=v2.InterpolationMode.NEAREST,  # preserves crisp edges
            fill=0.0,  # background stays black
        ),
        # Guardrail: keep size consistent (RandomAffine preserves size, but safe)
        v2.Resize((out_size, out_size), interpolation=v2.InterpolationMode.NEAREST),
        # keep silhouettes clean and binary-like
        v2.Lambda(lambda x: (x > 0.5).to(x.dtype)),
    ]

    if use_reflection:
        # Reflections are a discrete extension (O(2))—optional.
        tfms.insert(2, v2.RandomHorizontalFlip(p=0.5))
        tfms.insert(3, v2.RandomVerticalFlip(p=0.5))

    return v2.Compose(tfms)


def dsprites_augmentations(aug, img_size, adv=8 / 255):
    def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
        # s is the strength of color distortion
        return v2.RandomApply(
            [v2.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8
        )

    def stronger_distortion():
        # Check Fig. 10 of https://arxiv.org/pdf/2203.13457
        color_jitter = v2.ColorJitter(1.0, 1.0, 1.0, 0.5)
        return v2.RandomApply([color_jitter], p=0.8)

    if aug == "none":
        augmentations = [v2.Identity()]
    elif aug == "crop":
        augmentations = [
            v2.RandomResizedCrop(
                img_size, scale=(0.08, 1.0), interpolation=v2.InterpolationMode.NEAREST
            ),
            v2.Lambda(lambda x: (x > 0.5).to(x.dtype)),
        ]
    elif aug == "sup":
        augmentations = [
            v2.RandomResizedCrop(
                img_size, scale=(0.08, 1.0), interpolation=v2.InterpolationMode.NEAREST
            ),
            v2.Lambda(lambda x: (x > 0.5).to(x.dtype)),
            v2.RandomHorizontalFlip(),
        ]
    elif aug == "simclr":
        augmentations = [
            v2.RandomResizedCrop(
                img_size, scale=(0.08, 1.0), interpolation=v2.InterpolationMode.NEAREST
            ),
            v2.Lambda(lambda x: (x > 0.5).to(x.dtype)),
            v2.RandomHorizontalFlip(),
            get_color_distortion(s=0.5),
        ]

    elif aug == "simclr2":
        augmentations = [
            v2.RandomResizedCrop(
                img_size, scale=(0.08, 1.0), interpolation=v2.InterpolationMode.NEAREST
            ),
            v2.Lambda(lambda x: (x > 0.5).to(x.dtype)),
            v2.RandomHorizontalFlip(),
            get_color_distortion(s=1.0),
        ]
    elif aug == "simclr3":
        augmentations = [
            v2.RandomResizedCrop(
                img_size, scale=(0.08, 1.0), interpolation=v2.InterpolationMode.NEAREST
            ),
            v2.Lambda(lambda x: (x > 0.5).to(x.dtype)),
            v2.RandomHorizontalFlip(),
            stronger_distortion(),
        ]
    elif aug == "geom":
        # Example usage:
        augmentations = [
            dsprites_symmetries(
                out_size=img_size,
                max_translate_frac=0.30,
                max_rotate_deg=180.0,
                use_reflection=True,  # set True if you want reflection invariance
            )
        ]
    elif aug == "geom_crop":
        # Example usage:
        augmentations = [
            v2.RandomResizedCrop(
                img_size, scale=(0.08, 1.0)
            ),  # the interpolation and thresholding is done inside dsprites_symmetries
            dsprites_symmetries(
                out_size=img_size,
                max_translate_frac=0.30,
                max_rotate_deg=180.0,
                use_reflection=True,  # set True if you want reflection invariance
            ),
        ]
    adv_augmentations = augmentations.copy()
    adv_augmentations.insert(
        0, v2.Lambda(lambda x: x + torch.empty_like(x).uniform_(-adv, adv))
    )
    train_augmentations = v2.Compose(augmentations)
    adv_augmentations = v2.Compose(adv_augmentations)
    return train_augmentations, adv_augmentations
