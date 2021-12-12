"""Script for obtaining data for MNIST and SVHN."""
import numpy as np
import pathlib
from PIL import Image
import torch
import typing

from torchvision import transforms
from torchvision.datasets import MNIST, SVHN
from torch.utils.data import Dataset



# Directory constants.
THIS_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = THIS_DIR / "data"

def get_datasets(
    is_training: bool = False,
    size: int = 32,
) -> typing.Tuple[Dataset]:
    """Get datasets.

    Args:
        is_training: whether to use datasets for training
            or not.
        size: the size of the image after resizing.

    Returns:
        two datasets with RGB images of size 32x32,
        pixel values are in range [0, 1].
        Possible labels are {0, 1, 2, ..., 9}.
    """
    svhn = SVHN(
        DATA_DIR /'svhn',
        split='train' if is_training else 'test',
        download=True,
        transform=transforms.ToTensor(),
    )
    mnist_transform = transforms.Compose([
        # randomly color digit and background:
        transforms.Lambda(to_random_rgb),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    mnist = MNIST(
        DATA_DIR /'mnist',
        train=is_training,
        download=True,
        transform=mnist_transform,
    )
    return svhn, mnist


def to_random_rgb(x: torch.Tensor):
    color1 = np.random.randint(0, 256, size=3, dtype='uint8')
    color2 = np.random.randint(0, 256, size=3, dtype='uint8')
    x = np.array(x)
    x = x.astype('float32')/255.0
    x = np.expand_dims(x, 2)
    x = (1.0 - x) * color1 + x * color2
    return Image.fromarray(x.astype('uint8'))
