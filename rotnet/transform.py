import os
import random
from pathlib import Path
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Type,
                    Union)

import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate


class NCropAugmentation:
    def __init__(
        self, transform: Union[Callable, Sequence], num_crops: Optional[int] = None
    ):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Union[Callable, Sequence]): transformation pipeline or list of
                transformation pipelines.
            num_crops: if transformation pipeline is not a list, applies the same
                pipeline num_crops times, if it is a list, this is ignored and each
                element of the list is applied once.
        """

        self.transform = transform

        if isinstance(transform, Iterable):
            self.one_transform_per_crop = True
            assert num_crops == len(transform)
        else:
            self.one_transform_per_crop = False
            self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        if self.one_transform_per_crop:
            return [transform(x) for transform in self.transform]
        else:
            return [self.transform(x) for _ in range(self.num_crops)]


def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class TargetTransform:
    def __init__(self, n_class) -> None:
        self.n_class = n_class

    def __call__(self, label: int) -> Any:
        y = torch.zeros(self.n_class, dtype=torch.float32)
        y[label] = 1.0
        return y


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = [0.1, 2.0]):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gaussian blur to an input image.

        Args:
            x (torch.Tensor): an image in the tensor format.

        Returns:
            torch.Tensor: returns a blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)


class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)


class RotNetTransform:
    def __init__(self, num_rot: int = 4, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0) -> None:
        super().__init__()

        # Backward compatibility with integer value

        self.center = center

        self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0

        self.fill = fill

        self.num_rot = num_rot
        self.angles = [180/(i+1) for i in range(num_rot)]

    def __call__(self, x: Image) -> Any:
        index = torch.randperm(self.num_rot)[0]
        angle = self.angles[index]
        transforms.RandomRotation()
        return angle, rotate(
            x, angle, self.interpolation, self.expand, self.center, self.fill
        )


class CifarTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.0,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
    ):
        """Applies cifar transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
        """

        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (32, 32),
                    scale=(min_scale, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness, contrast, saturation, hue)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )


class STLTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.0,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
    ):
        """Applies STL10 transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (96, 96),
                    scale=(min_scale, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness, contrast, saturation, hue)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )
