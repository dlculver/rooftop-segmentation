"""
Script for forming a Dataset from Dida data
and performing necessary preprocessing (e.g. normalization)
"""

# import necessary libraries
# import os
from pathlib import Path

import numpy as np
from PIL import Image

# import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# import torchvision.transforms as transforms
from torchvision.transforms import functional as F


# helper functions
def open_image(path, mode=None):
    """
    opens image from directory
    """
    img = Image.open(path, mode="r")
    img.load()
    if (
        mode and img.mode != mode
    ):  # in case we need to make something in gray scale for exmaple
        img = img.convert(mode)
    return img


class RoofDataSet(Dataset):
    """
    Organizes Dida dataset into a Dataset class

    Parameters:
        data_dir : (str) directory which contains the dataset
        mode : (str) can be 'train', 'val', or 'test'. Defaults to 'train'
        normalize : (bool) determines whether we normalize image tensors
        transform : (bool) determines if we perform data augmentaiton
        resize : determines if we allow for resizing of images
    """

    img_mean = [0.485, 0.456, 0.406]  # from pretrained MobileNet
    img_std = [0.229, 0.224, 0.225]  # from pretrained MobileNet

    def __init__(
        self,
        data_dir,
        mode="train",
        normalize=True,
        # no_labels=False,
        transform=False,
        resize=None,
        binary_mask=False,
    ):
        self.mode = mode
        self.normalize = normalize
        self.resize = resize

        self.images = self.load_images(Path(f"{data_dir}/{mode}/images"), mode="RGB")
        self.labels = (
            self.load_images(Path(f"{data_dir}/{mode}/labels"), mode="L")
            if mode != "test"
            else None
        )
        self.binary_mask = binary_mask

        self.transform = (
            RandomSubset(
                [
                    RandomSimultaneousHorizontalFlip(),
                    RandomSimultaneousVerticalFlip(),
                    RandomSimultaneousResizeCrop(
                        256, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)
                    ),
                    RandomSimultaneousRotation(180),
                ]
            )
            if transform
            else None
        )

    @staticmethod
    def load_images(dir_: Path, mode=None):
        """
        method for loading images from directory
        """
        paths = sorted(
            [p for p in dir_.iterdir() if p.is_file() and p.name != "278.png"]
        )  # there is a mismatch between image and label for 278.png, skip over to help with training
        images = [open_image(p, mode) for p in paths]
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx] if self.labels else None

        if self.resize:
            img = F.resize(img, self.resize)
            label = F.resize(label, self.resize)
        if self.transform:
            img, label = self.transform([img, label])
        img = F.to_tensor(img)
        if self.normalize:
            img = F.normalize(img, self.img_mean, self.img_std)
        if label:
            label = F.to_tensor(label)
            if self.binary_mask:
                label = (label > 0.5).float()
            return img, label
        else:
            return img

    def inverse_normalize(self, img):
        img = img.clone()
        for i in range(3):
            img[i] = img[i] * self.img_std[i] + self.img_mean[i]
        return img


# we need to allow for data augmentation, transforms must apply synchronously to bouth image and label


class RandomSubset:
    """
    Class to apply a random subset of our transforms to the images
    """

    def __init__(self, transforms: list):
        self.transforms = transforms  # transforms is a list

    def get_rand_subset(self):
        # randomly draw how many transforms to select
        num = np.random.randint(0, len(self.transforms))
        # draw which transforms to use
        subset = np.random.choice(self.transforms, num, replace=False)
        return subset

    def __call__(self, img):
        subset = self.get_rand_subset()
        for t in subset:
            img = t(img)

        return img


class RandomSimultaneousHorizontalFlip:
    """
    Class for simultaneously horizontally flipping a list of images
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images: list):
        if np.random.random() < self.prob:
            for i in range(len(images)):
                images[i] = F.hflip(images[i])

        return images


class RandomSimultaneousVerticalFlip:
    """
    Class for simultaneously vertically flipping a list of images
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images: list):
        if np.random.random() < self.prob:
            for i in range(len(images)):
                images[i] = F.vflip(images[i])

        return images


class RandomSimultaneousResizeCrop:
    """
    Applies a crop and resize operation randomly and simultaneously to a list of images
    """

    def __init__(self, size, scale, ratio, interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    @staticmethod
    def get_size(image, scale, ratio):
        for _ in range(10):
            area = image.size[0] * image.size[1]
            target_area = np.random.uniform(*scale) * area  # get scaled area
            aspect_ratio = np.random.uniform(*ratio)  # get aspect ratio

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(
                round(np.sqrt(target_area / aspect_ratio))
            )  # so area is still target_area

            if np.random.random() < 0.5:
                w, h = h, w

            if w < image.size[0] and h < image.size[1]:
                i = np.random.randint(0, image.size[1] - h)
                j = np.random.randint(0, image.size[0] - w)
                return i, j, h, w

        w = min(image.size[0], image.size[1])
        i = (image.size[1] - w) // 2
        j = (image.size[0] - w) // 2

        return i, j, w, w

    def __call__(self, imgs):
        # assume images are all the same size
        # resize and crop images in list synchronously
        i, j, h, w = self.get_size(imgs[0], self.scale, self.ratio)
        for i in range(len(imgs)):
            imgs[i] = F.resized_crop(imgs[i], i, j, h, w, self.size, self.interpolation)
        return imgs


class RandomSimultaneousRotation:
    """
    Randomly apply a rotation synchronously to a list of images
    """

    def __init__(self, degree_range, resample=False, expand=False, center=None):
        if isinstance(degree_range, int):
            if degree_range < 0:
                raise ValueError(
                    "If degree_range is a single number it must be positive."
                )
            self.degree_range = (-degree_range, degree_range)
        else:
            if len(degree_range) != 2:
                raise ValueError("If degree_range is a list, it must have length 2.")
            self.degree_range = degree_range

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_theta(degree_range):
        return np.random.uniform(degree_range)

    def __call__(self, images):
        theta = self.get_theta(self.degree_range)[0]
        for i in range(len(images)):
            images[i] = F.rotate(
                images[i], theta, self.resample, self.expand, self.center
            )
        return images
