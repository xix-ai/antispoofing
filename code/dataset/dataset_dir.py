import glob
import pickle as pkl

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .randaugment_loc import RandAugmentMC

size = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB

transform_train = A.Compose(
    [
        A.RandomResizedCrop(size, size, scale=(0.33, 1.0), ratio=(0.7, 1.35)),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.2
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3
        ),
        A.CoarseDropout(
            max_height=24, max_width=24, p=0.15
        ),
        A.RandomRotate90(p=0.3),
        A.GaussNoise(var_limit=(10.0, 20.0), p=0.15),
        A.MotionBlur(blur_limit=[3, 8], p=0.15),
        A.Downscale(scale_min=0.25, scale_max=0.5, p=0.15),
        A.ISONoise(p=0.15),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
    ]
)
transform_weak = A.Compose(
    [
        A.RandomResizedCrop(size, size, scale=(0.7, 0.9), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
    ]
)

transform_test = A.Compose(
    [
        A.Resize(size, size),
    ]
)
transform_tests = [
    A.Compose([
        A.SmallestMaxSize(max_size=int(1.2 * size), p=1.0),
        A.CenterCrop(size, size),
    ]),
    A.Compose([
        A.SmallestMaxSize(max_size=int(1.1 * size), p=1.0),
        A.CenterCrop(size, size),
        A.HorizontalFlip(p=1),

    ]),
    A.Compose([
        A.Resize(size, size),

    ]),
]


def normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD, value=255.0):
    image = image.astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = image / value
    image = (image - mean) / std
    return image


class DatasetDir(Dataset):
    """Face Landmarks dataset."""

    def __init__(
            self,
            data_dir,
            pkl_path,
            idx=None,
            transform=transform_train,
            state="train",
            tensor_norm=False,
            is_add=False,
    ):
        self.state = state
        self.transform = transform
        self.data_dir = data_dir
        self.tensor_norm = tensor_norm
        self.idx = idx
        self.data = pkl.load(open(pkl_path, 'rb'))
        self.list_dataset = isinstance(self.data, list)
        self.is_add = is_add
        if self.list_dataset:
            print('LIST dataset')
            self.names = self.data[1]
            self.attrs = self.data[0]
            self.n_attrs = len(self.attrs)
        else:
            print('DICT dataset')
            self.attrs = self.data
            self.names = list(self.attrs.keys())
            self.n_attrs = len(self.attrs[self.names[0]])

        if idx is None:
            self.idx = np.arange(len(self.names))
        if state == 'test':
            self.transform = transform_test

    def __getitem__(self, image_index):
        """
        :return: Tuple (
            image: torch.Tensor[3, size, size],
            attributes: torch.Tensor(self.nattrs, ),
            path: str
        )
        """
        try:
            index = self.idx[image_index]
            path = f'{self.data_dir}/{self.names[index]}'
            image = cv2.imread(path)
        except:
            print(index, image_index)
            print(f"Error loading image with index {image_index}: mapped to {index}")
            return torch.zeros((3, size, size)).float(), np.zeros(self.n_attrs, np.float32), ''
        if image is None:
            print(f"Image not found: {path}")
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = self.to_tensor(image)

        if self.list_dataset:
            attrs = self.attrs[index]
        else:
            attrs = self.attrs[self.names[index]]
        if self.is_add:
            attrs = attrs.tolist() + [0, 1]
            if np.sum(attrs) <= 1:
                attrs[-2] = 1
        return image, np.array(attrs), path

    def to_tensor(self, x):
        if self.tensor_norm:
            x = normalize(x)
        elif x.dtype == np.uint8:
            x = x / 255
        x = x.transpose(2, 0, 1)
        return torch.from_numpy(x).float()

    def __len__(self):
        return len(self.idx)


class DatasetImage(Dataset):
    """Face Landmarks dataset."""

    def __init__(
            self,
            data_dir,
            transform=transform_train,
            is_test=False,
            is_multi=False,
            tensor_norm=False,
    ):

        self.data_dir = data_dir
        self.paths = glob.glob(f"{data_dir}/**/*.jpg")
        self.paths += glob.glob(f"{data_dir}/**/*.jpeg")
        self.paths += glob.glob(f"{data_dir}/*.jpg")
        if is_test:
            self.transform = transform_tests
        else:
            self.transform = transform_train
        self.is_multi = is_multi
        self.is_test = is_test
        self.tensor_norm = tensor_norm

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(path)
        key = path.split('/')[-1].split('.')[0]
        if not self.is_multi:
            if self.transform is not None:
                image = transform_test(image=image)["image"]
            image = self.to_tensor(image)
            return image, path
        else:
            images = []
            for transform in self.transform:
                image_cur = transform(image=image)["image"]
                images.append(self.to_tensor(image_cur))
            return images, path

    def to_tensor(self, x):
        if self.tensor_norm:
            x = normalize(x)
        elif x.dtype == np.uint8:
            x = x / 255
        x = x.transpose(2, 0, 1)

        return torch.from_numpy(x).float()

    def __len__(self):
        return len(self.paths)


class FixMatchImage(Dataset):
    """Face Landmarks dataset."""

    def __init__(
            self,
            data_dir,
            is_test=False,
            is_multi=False
    ):

        self.data_dir = data_dir
        self.paths = glob.glob(f"{data_dir}/**/*.jpg")
        self.paths += glob.glob(f"{data_dir}/**/*.jpeg")
        self.paths += glob.glob(f"{data_dir}/*.jpg")
        self.RA = RandAugmentMC(n=2, m=10, size=size)

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(path)
        key = path.split('/')[-1].split('.')[0]

        image_w = transform_weak(image=image)["image"]
        image_w = self.to_tensor(image_w)
        image_s = transform_train(image=image)["image"]
        image_s = self.to_tensor(image_s)
        return image_w, image_s, path

    def to_tensor(self, x):
        x = x.transpose(2, 0, 1)
        if x.dtype == np.uint8:
            x = x / 255
        return torch.from_numpy(x).float()

    def __len__(self):
        return len(self.paths)
