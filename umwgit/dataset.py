from functools import partial
import pickle
from types import FunctionType
from typing import Iterable
import ast
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from umwgit.transforms import train_transforms, valid_transforms


class TractDataset(Dataset):
    def __init__(self, df, label=True,
                 transforms: FunctionType = None):
        self.df = df
        self.label = label
        self.ids = df['id'].tolist()
        self.img_paths = df['image_paths'].tolist()
        if label:
            self.mask_paths = df['mask_path'].tolist()
        else:
            self.mask_paths = None
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        img_paths = ast.literal_eval(self.img_paths[index])
        img = load_imgs(img_paths)
        id_ = self.ids[index]
        h, w, c = img.shape
        if self.label:
            mask_path = self.mask_paths[index]
            mask = load_mask(mask_path)
            if self.transforms:
                tf = self.transforms(image=img, max_old_dim=max(h, w))
                data = tf(image=img, mask=mask)
                img = data['image']
                mask = data['mask']
            mask = np.transpose(mask, (2, 0, 1))
            return img, mask, h, w
        else:
            if self.transforms:
                tf = self.transforms(image=img, max_old_dim=max(h, w))
                img = tf(image=img)['image']
            return img, id_, h, w


def load_imgs(paths: Iterable[str]):
    img = [None for _ in paths]
    for i, path in enumerate(paths):
        if path != '':
            channel = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            channel = channel.astype('float32')  # original is uint16
            img[i] = channel
            h, w = channel.shape
    img = [channel if channel is not None else np.zeros(
        (h, w)) for channel in img]
    return np.stack(img, axis=-1).astype(np.float32)


def load_mask(path: str):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    mask = mask.astype('float32')
    return mask


def create_dataloader(path: str, df: pd.DataFrame, img_size: int, batch_size: int,
                      shuffle_valid: bool = False, train_tf: FunctionType = train_transforms,
                      valid_tf: FunctionType = valid_transforms):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    train_idx, val_idx = d['train_idx'], d['val_idx']
    train_set = TractDataset(
        df.loc[train_idx], transforms=partial(train_tf, new_dim=img_size))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_set = TractDataset(
        df.loc[val_idx], transforms=partial(valid_tf, new_dim=img_size))
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=shuffle_valid)

    return train_loader, valid_loader
