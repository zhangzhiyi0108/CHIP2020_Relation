import csv
import os
import time
from abc import ABC
from pathlib import Path

import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from base.runner.base_runner import BaseRunner


class CommonRunner(BaseRunner, ABC):
    """
    common implementation for runner
    """

    def __init__(self, runner_config):
        super(BaseRunner, self).__init__()

        self._config = runner_config
        self._device = self._config.device
        self._train_dataloader = None
        self._valid_dataloader = None
        self._model = None
        self._loss = None
        self._optimizer = None

        self._build()

        self._model_name = self._config.model.name + "_" + self._config.data.name

        #   for checkpoint
        dir_saved = self._config.learn.dir.saved
        Path(dir_saved).mkdir(parents=True, exist_ok=True)
        self._model_path = os.path.join(dir_saved, str(self._model_name + '.ckp'))

        #   for global_step
        self.global_step = 0

        #   for summary
        dir_summary = self._config.learn.dir.summary
        Path(dir_summary).mkdir(parents=True, exist_ok=True)
        self._summary_log_dir = os.path.join(dir_summary, self._model_name)

        #   for log
        dir_log = self._config.learn.dir.log
        Path(dir_log).mkdir(parents=True, exist_ok=True)
        self._valid_log_fields = None
        self._valid_log_filepath = os.path.join(dir_log, self._model_name + "_valid_log.csv")

        self._train_fmt = "train: episode={:4d}, global_step={:6d}, batch={:4d}, " \
                          "batch_size={:4d}, batch_loss={:.4f}, elapsed={:.4f}"
        pass

    def _build(self):
        self._build_data()
        self._build_model()
        self._build_loss()
        self._build_scheduler()

    def _build_scheduler(self):
        self._optimizer = torch.optim.Adam(
            params=self._model.parameters(),
            lr=self._config.learn.learning_rate,
            weight_decay=self._config.learn.weight_decay
        )
        self._scheduler = StepLR(self._optimizer, step_size=2000, gamma=0.1)
        pass

    def train(self):
        # switch to train mode
        self._model.train()

        print("training...")
        with SummaryWriter(logdir=self._summary_log_dir, comment='model') as summary_writer, \
                open(self._valid_log_filepath, mode='w') as valid_log_file:
            valid_log_writer = csv.writer(valid_log_file, delimiter=',')
            valid_log_writer.writerow(self._valid_log_fields)
            for episode in range(self._config.learn.episode):
                self._train_epoch(episode, summary_writer)
                self._valid(episode, valid_log_writer, summary_writer)
                self._display_result(episode)
        pass

    def _train_epoch(self, episode, summary_writer):
        epoch_start = time.time()
        self._model.train()
        batch = 0
        for dict_input in tqdm(self._train_dataloader):

            dict_output = self._model(dict_input)
            dict_loss = self._loss(dict_output)

            # Backward and optimize
            self._optimizer.zero_grad()  # clear gradients for this training step
            batch_loss = dict_loss['loss_batch']
            batch_loss.backward()  # back-propagation, compute gradients
            self._optimizer.step()  # apply gradients

            self.global_step += 1
            batch += 1
            if self.global_step % self._config.learn.batch_display == 0:
                for loss_key, loss_value in dict_loss.items():
                    summary_writer.add_scalar('loss/' + loss_key, loss_value, self.global_step)
                # summary_writer.flush()
                elapsed = time.time() - epoch_start
                print(self._train_fmt.format(
                    episode + 1, self.global_step, batch,
                    self._config.data.train_batch_size,
                    batch_loss.item(), elapsed
                ))
                epoch_start = time.time()
        pass

    def _save_checkpoint(self, epoch):
        torch.save({
            # 'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
        }, self._model_path)
        pass

    def _load_checkpoint(self):
        config = Path(self._model_path)
        if config.is_file():
            print("loading saved pretrained model from {}.".format(self._model_path))
            checkpoint = torch.load(self._model_path)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epoch = checkpoint['epoch']
            # loss = checkpoint['loss']
        else:
            print("No model exists in {}.".format(self._model_path))

        self._model.to(self._config.device)
        return
