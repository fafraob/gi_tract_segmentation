from math import ceil
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def train_transforms(image: np.ndarray, new_dim: int, max_old_dim: int):
    return A.Compose([
        *size_transforms_list(new_dim, max_old_dim),
        A.Rotate(
            limit=(0, 359),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.75
        ),
        A.RandomResizedCrop(
            height=new_dim,
            width=new_dim,
            scale=(0.2, 0.8),
            ratio=(1, 1),
            interpolation=cv2.INTER_LINEAR,
            p=0.33
        ),
        A.CoarseDropout(
            max_holes=7,
            max_height=ceil(new_dim*0.1),
            max_width=ceil(new_dim*0.1),
            fill_value=0,
            mask_fill_value=None,
            p=0.5
        ),
        *norm_transforms_list(image)
    ])


def valid_transforms(image: np.ndarray, new_dim: int, max_old_dim: int):
    return A.Compose([
        *size_transforms_list(new_dim, max_old_dim),
        A.Rotate(
            limit=(0, 359),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.75
        ),
        *norm_transforms_list(image)
    ])


def test_transforms(image: np.ndarray, new_dim: int, max_old_dim: int):
    return A.Compose([
        *size_transforms_list(new_dim, max_old_dim),
        *norm_transforms_list(image)
    ])


def reshape_transforms(old_h: int, old_w: int):
    max_side = max(old_h, old_w)
    return A.Compose([
        A.Resize(
            max_side, max_side,
            interpolation=cv2.INTER_AREA,
            p=1.0
        ),
        A.Crop(
            x_max=old_w,
            y_max=old_h,
            p=1.0
        )
    ], p=1.0)


def size_transforms_list(new_dim: int, max_old_dim: int):
    return [
        A.PadIfNeeded(
            min_width=max_old_dim,
            min_height=max_old_dim,
            position=A.PadIfNeeded.PositionType.TOP_LEFT,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0
        ),
        A.Resize(
            height=new_dim,
            width=new_dim,
            interpolation=cv2.INTER_CUBIC,
            p=1.0
        )
    ]


def norm_transforms_list(image: np.ndarray):
    channels = image.shape[2]
    return [
        A.Normalize(
            mean=[np.mean(image[:, :, c]) for c in range(channels)],
            std=[np.std(image[:, :, c]) +
                 1e-6 for c in range(channels)],
            max_pixel_value=1
        ),
        ToTensorV2()
    ]
