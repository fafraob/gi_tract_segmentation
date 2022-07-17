import pandas as pd
from umwgit.dataset import create_dataloader
from umwgit.trainer import Trainer
from umwgit.model import build_ssformer, load_ssformer
from torch.optim import AdamW
from umwgit.utils import criterion, dice_coef, PolynomialLRDecay, parse_args


def get_optim(model, args):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = PolynomialLRDecay(
        optimizer,
        max_decay_steps=args.max_decay_steps,
        end_learning_rate=args.end_lr,
        power=args.lr_power
    )
    return optimizer, scheduler


def main(args):
    if args.weights:
        model = load_ssformer(args.weights)
    else:
        model = build_ssformer()
    optimizer, scheduler = get_optim(model, args)
    # data
    df = pd.read_csv('data/data.csv')
    train_loader, valid_loader = create_dataloader(
        f'data/folds/fold_{args.fold}.pickle', df, img_size=args.image_size, batch_size=args.batch_size)
    # trainer
    trainer = Trainer(
        name=args.name,
        model=model,
        loss_func=criterion,
        max_epochs=args.epochs,
        score_func=dice_coef,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device
    )
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    trainer.fit(train_loader, valid_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
