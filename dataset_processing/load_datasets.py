import numpy as np
import torch
from torch.nn import init
from torchvision.transforms import v2


class DietGrayDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=[0.5], std=[0.5]),
        augmentations=None,
    ):
        super().__init__()
        self.images = images
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    augmentations,
                    normalize,
                ]
            )

    def __getitem__(self, idx):
        x = self.images[idx]
        return self.transform(x), idx

    def __len__(self):
        return len(self.images)


class GrayDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=[0.5], std=[0.5]),
        augmentations=None,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    augmentations,
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)
        return x, y


class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=[0.5], std=[0.5]),
        augmentations=None,
        nch=1,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentations = augmentations
        normalize = v2.Normalize(mean=[0.5] * nch, std=[0.5] * nch)
        if augmentations is None:
            self.transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    augmentations,
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        x_1 = self.transform(x)
        x_2 = self.transform(x)
        return x_1, x_2, y


class RGBContrastiveDataset(ContrastiveDataset):
    def __init__(self, images, labels, augmentations=None):
        # Call the parent constructor and explicitly set nch=3
        super().__init__(
            images=images, labels=labels, augmentations=augmentations, nch=3
        )


class BeforeAttack(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=None,
        resize=None,
        augmentations=None,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentations = augmentations
        transform_ = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        if resize is not None:
            transform_.append(v2.Resize(resize))
        if augmentations is not None:
            transform_.append(augmentations)
        if normalize is not None:
            transform_.append(normalize)
        self.transform = v2.Compose(transform_)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)
        return x, y


class RGBBeforeAttack(BeforeAttack):
    def __init__(
        self,
        images,
        labels,
        normalize=None,
        resize=None,
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


# TODO make abstract class that works on all datasets
class DietDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=[0.5], std=[0.5]),
        augmentations=None,
    ):
        super().__init__()
        self.images = images
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=False), normalize]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                    augmentations,
                    normalize,
                ]
            )

    def __getitem__(self, idx):
        x = self.images[idx]
        return self.transform(x), idx

    def __len__(self):
        return len(self.images)


class DislibDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=[0.5], std=[0.5]),
        augmentations=None,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=False), normalize]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                    augmentations,
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)
        return x, y


class ContrastiveDislibDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=[0.5], std=[0.5]),
        augmentations=None,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=False), normalize]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                    augmentations,
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        x_1 = self.transform(x)
        x_2 = self.transform(x)
        return x_1, x_2, y


class ContrastiveRGBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        augmentations=None,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    normalize,
                ]  # [0, 255] -> [0., 1.]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    augmentations,
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        x_1 = self.transform(x)
        x_2 = self.transform(x)
        return x_1, x_2, y


class RGBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        augmentations=None,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    normalize,
                ]  # [0, 255] -> [0., 1.]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    augmentations,
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)
        return x, y


class DietRGBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        augmentations=None,
    ):
        super().__init__()
        self.images = images
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    normalize,
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    augmentations,
                    normalize,
                ]
            )

    def __getitem__(self, idx):
        x = self.images[idx]
        return self.transform(x), idx

    def __len__(self):
        return len(self.images)


class MPI3DDataset(RGBDataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        augmentations=None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                    normalize,
                ]  # [0, 255] -> [0., 1.]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                    augmentations,
                    normalize,
                ]
            )


class DietMPI3DDataset(DietRGBDataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        augmentations=None,
    ):
        self.images = images
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                    normalize,
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                    augmentations,
                    normalize,
                ]
            )
