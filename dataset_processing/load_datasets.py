import numpy as np
import torch
from torchvision.transforms import v2


# TODO make abstract class that works on all datasets
class DietDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.ToDtype(torch.float32, scale=False),
        augmentations=None,
    ):
        super().__init__()
        self.images = images
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]
            )
        else:
            self.transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=False), augmentations]
            )

    def __getitem__(self, idx):
        x = self.images[idx]
        return self.transform(x), idx

    def __len__(self):
        return len(self.images)


class DietRGBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Lambda(lambda x: x / 255.0),
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
        normalize=v2.ToDtype(torch.float32, scale=False),
        augmentations=None,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentations = augmentations
        if augmentations is None:
            self.transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                    augmentations,
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


class RGBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        normalize=v2.Lambda(lambda x: x / 255.0),
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
