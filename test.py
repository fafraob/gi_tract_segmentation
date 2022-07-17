from functools import partial
from matplotlib import pyplot as plt
import pandas as pd
import torch
from umwgit.dataset import create_dataloader
from umwgit.transforms import reshape_transforms
from umwgit.utils import criterion, dice_coef, predict_tta, parse_args
from umwgit.model import load_ssformer
from umwgit.utils import PolynomialLRDecay
from umwgit.trainer import Trainer
import numpy as np


def prep_img(image):
    img = image
    for c in range(3):
        img[:, :, c] -= np.min(img[:, :, c])
        mx = np.max(img[:, :, c])
        if mx:
            img[:, :, c] /= mx
    return img


def compare_masks(images, preds, masks):
    size = len(images)
    fig, axs = plt.subplots(size, 3, figsize=(7, 12))
    for i, (img, pred, msk) in enumerate(zip(images, preds, masks)):
        img = prep_img(img)
        axs[i, 0].imshow(img, cmap='bone')
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 1].imshow(pred*255, cmap='bone')
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])
        axs[i, 2].imshow(msk*255, cmap='bone')
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig('test.png')


def lr_scheduler_test(args):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = PolynomialLRDecay(
        optimizer,
        max_decay_steps=args.max_decay_steps,
        end_learning_rate=args.end_lr,
        power=args.lr_power
    )
    lrs = []

    for i in range(args.epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    plt.plot(lrs)
    plt.savefig('lrs.png')
    print(lrs)


@torch.no_grad()
def infer_test(args):
    df = pd.read_csv('data/data.csv')
    _, loader = create_dataloader(
        f'data/folds/fold_{args.fold}.pickle', df, img_size=args.image_size,
        batch_size=args.batch_size, shuffle_valid=True
    )
    imgs, msks, h, w = next(iter(loader))
    model = load_ssformer(args.weights)
    model.to('cuda')
    outs = predict_tta(model, imgs)
    msks = msks.to('cuda')
    for i in range(args.batch_size):
        print(dice_coef(outs[i].unsqueeze(0), msks[i].unsqueeze(0)).item())
    outs = outs.to('cpu')
    imgs = imgs.to('cpu')
    msks = msks.to('cpu')
    tfs = [reshape_transforms(int(old_h), int(old_w))
           for old_h, old_w in zip(h, w)]
    imgs = [tf(image=img.permute((1, 2, 0)).detach().numpy())['image']
            for tf, img in zip(tfs, imgs)]
    msks = [tf(image=msk.permute((1, 2, 0)).detach().numpy())['image']
            for tf, msk in zip(tfs, msks)]
    outs = [(tf(image=out.permute((1, 2, 0)).detach().numpy())
             ['image'] > 0.5) for tf, out in zip(tfs, outs)]
    compare_masks(imgs, outs, msks)


def eval_test(args):
    model = load_ssformer(args.weights)
    df = pd.read_csv('data/data.csv')
    df = df[df.n_segs == 0]
    _, valid_loader = create_dataloader(
        f'data/folds/fold_{args.fold}.pickle', df, img_size=args.image_size, batch_size=args.batch_size, shuffle_valid=False)
    trainer = Trainer(
        name='test',
        model=model,
        loss_func=criterion,
        score_func=partial(dice_coef, thr=args.thr),
        device=args.device
    )
    trainer.eval(valid_loader)


if __name__ == '__main__':
    args = parse_args()
    import time
    start_time = time.time()
    infer_test(args)
    print("--- %s seconds ---" % (time.time() - start_time))
