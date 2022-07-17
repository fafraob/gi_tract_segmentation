import gc
import time
from types import FunctionType
import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class Trainer():

    def __init__(self, name: str, model: nn.Module, loss_func: FunctionType,
                 score_func: FunctionType, optimizer: optim.Optimizer = None,
                 scheduler: optim.lr_scheduler._LRScheduler = None,
                 max_epochs: int = 100, device: str = 'cuda'):
        super().__init__()
        self.name = name
        self.model = model
        self.loss_func = loss_func
        self.score_func = score_func
        self.device = device
        self.model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self._curr_epoch = 0
        self._best_val_score = -np.inf
        self._log_path = self.name + '_log'

    def _batch_step(self, batch: torch.tensor):
        x, y, h, w = batch
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        y_hat = torch.nn.Sigmoid()(y_hat)
        score = self.score_func(y_hat, y)
        return loss, score

    def _run_epoch(self, data_loader: DataLoader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        progress_bar = tqdm(data_loader, total=len(
            data_loader), desc='Train' if train else 'Valid')
        dataset_size, running_loss, running_score = 0, 0, 0
        for batch in progress_bar:
            if train:
                self.optimizer.zero_grad()
                loss, score = self._batch_step(batch)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    loss, score = self._batch_step(batch)
            batch_size = batch[0].size(0)
            running_loss += (loss.item() * batch_size)
            running_score += (score.item() * batch_size)
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            epoch_score = running_score / dataset_size
            progress_bar.set_postfix(
                loss=f'{epoch_loss:0.5f}',
                score=f'{epoch_score:0.5f}'
            )
        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss, epoch_score

    def _keep_running(self):
        if self._curr_epoch >= self.max_epochs:
            return False
        return True

    def _eval_val_score(self):
        if self.val_score >= self._best_val_score:
            print(('Valid Score Improved '
                   f'({self._best_val_score:0.5f} ---> '
                   f'{self.val_score:0.5f})'))
            self._best_val_score = self.val_score
            torch.save(self.model.state_dict(), f'{self.name}.pth')

    def _print_scores(self):
        s = (f'Epoch {self._curr_epoch}: '
             f'Train Score: {self.train_score:0.5f} | '
             f'Valid Score: {self.val_score:0.5f}')
        print(s)

    def _print_elapsed(self):
        time_elapsed = time.time() - self.start_time
        print((f'Training complete in {time_elapsed // 3600:0.0f}h '
               f'{(time_elapsed % 3600) // 60:0.0f}m '
               f'{(time_elapsed % 3600) % 60:0.0f}s'))
        print(f'Best Score: {self._best_val_score:0.5f}')

    def _log_scores(self):
        self.logger.add_scalars(
            'loss',
            {'train': self.train_loss, 'val': self.val_loss},
            self._curr_epoch
        )
        self.logger.add_scalars(
            'score',
            {'train': self.train_score, 'val': self.val_score},
            self._curr_epoch
        )

    def eval(self, val_loader: DataLoader):
        _, val_score = self._run_epoch(val_loader, train=False)
        print(f'Valid Score: {val_score:0.5f}')

    def _save_checkpoint(self):
        torch.save({
            'epoch': self._curr_epoch,
            'model_params': self.model.state_dict(),
            'optimizer_params': self.optimizer.state_dict(),
            'scheduler_params': self.scheduler.state_dict(),
            'log_path': self._log_path,
            'best_score': self._best_val_score

        }, f'{self.name}_ckpt.pth')

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self._curr_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_params'])
        self.optimizer.load_state_dict(checkpoint['optimizer_params'])
        self.scheduler.load_state_dict(checkpoint['scheduler_params'])
        self._best_val_score = checkpoint['best_score']
        self._log_path = checkpoint['log_path']
        print('Checkpoint loaded.')

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        self.start_time = time.time()
        self.logger = SummaryWriter(log_dir=self._log_path)
        while self._keep_running():
            self._curr_epoch += 1
            self.train_loss, self.train_score = self._run_epoch(
                train_loader, train=True)
            self.val_loss, self.val_score = self._run_epoch(
                val_loader, train=False)
            if self.scheduler:
                self.scheduler.step()
            self._log_scores()
            self._print_scores()
            self._eval_val_score()
            self._save_checkpoint()
        self.logger.close()
        self._print_elapsed()
