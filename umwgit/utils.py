import argparse
from pyexpat import model
from torchvision.transforms.functional import hflip
from torch.optim.lr_scheduler import _LRScheduler
from segmentation_models_pytorch.losses import (
    DiceLoss, SoftBCEWithLogitsLoss)
import torch
import numpy as np
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='model',
                        help='Name that is used by the trainer to save logs, weights, etc.')
    parser.add_argument(
        '--fold', default=0, help='Fold to use for training. Everything should be prepared during data preparation.')
    parser.add_argument('--weights', default='',
                        help='Specify weights you want to use for your model.')
    parser.add_argument('--checkpoint', default='',
                        help='Specify the path to the checkpoint file to continue training from a checkpoint.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--end_lr', type=float, default=1e-7,
                        help='Final learning rate of the polynomial learning rate scheduler.')
    parser.add_argument('--max_decay_steps', type=int, default=30,
                        help='After the specified amount of epochs, "end_lr" is used for the remaining epochs.')
    parser.add_argument('--lr_power', type=float, default=1.0,
                        help='Learning rate power for the polynomial learning rate scheduler.')
    args = parser.parse_args()
    print(args)

    return args


def predict_tta(model: nn.Module, img_batch: torch.tensor,
                device: str = 'cuda') -> np.ndarray:
    """
    Predict with horizontal flip test time augmentation.
    """
    img_batch = img_batch.to(device, dtype=torch.float32)
    model.to(device)
    n, _, h, w = img_batch.size()
    mask = torch.zeros((n, 3, h, w), device=device, dtype=torch.float32)
    mask += model(img_batch) / torch.tensor(2.0)
    flipped_batch = hflip(img_batch)
    flipped_out = model(flipped_batch)
    out = hflip(flipped_out)
    mask += out / torch.tensor(2.0)

    return nn.Sigmoid()(mask)


DICELoss = DiceLoss(mode='multilabel')
BCELoss = SoftBCEWithLogitsLoss(reduction='mean')


def dice_coef(y_pred, y_true, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(axis=dim)
    den = y_true.sum(axis=dim) + y_pred.sum(axis=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(axis=(1, 0))
    return dice


def criterion(y_pred, y_true):
    return 0.5 * DICELoss(y_pred, y_true) + 0.5 * BCELoss(y_pred, y_true)


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of 
            learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr
