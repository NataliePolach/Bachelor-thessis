# %%
import torch
import cv2
import numpy as np
import torchvision.transforms as T

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self, norm_type=cv2.NORM_MINMAX):
        self.norm_type = norm_type

    def __call__(self, image, targets):

        # image = super().__call__(tensor)
        image = cv2.normalize(image, None, 0, 255, self.norm_type)
        return image, targets

class ToTensor(object):
    """Ndarray in sample to Tensor."""

    def __call__(self, image, targets):

        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image[None, :, :, :]).float()
        return image, targets

class Compose(object):
    """
        Composes transforms together."""


    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets):
        for t in self.transforms:
            image, targets = t(image, targets)
        return image, targets