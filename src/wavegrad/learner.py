# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
#from tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

from wavegrad.dataset import from_path as dataset_from_path
from wavegrad.dataset import from_file_lists as dataset_from_lists
from wavegrad.dataset import crop_features
from wavegrad.model import WaveGrad
from wavegrad.noise_schedule import get_noise_schedule

import sys
import signal

def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return { k: _nested_map(v, map_fn) for k, v in struct.items() }
    return map_fn(struct)

# sample from a, v, av
def sample(a_features, v_features, av_features, sample_probs):
    batch_size = a_features.shape[0]
    rn = np.random.rand(batch_size)

    features = []
    for i, r in enumerate(rn):
        if r < sample_probs[0]:
            features.append(a_features[i])
        elif r > (sample_probs[0] + sample_probs[1]):
            features.append(av_features[i])
        else:
            features.append(v_features[i])
    features = torch.stack(features, dim=0)
    return features

class SignalHandler:
    def __init__(self, learner):
        self.learner = learner

    def handle(self, sig, frames):
        print(f"learner {self.learner.replica_id} receive {sig}")
        if self.learner.is_master:
            print(f"save checkpoint for step {self.learner.step}")
            self.learner.save_to_checkpoint()
            exit(0)
            #raise KeyboardInterrupt


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class WaveGradLearner:
    def __init__(self, replica_id, model_dir, model, train_dataset, valid_dataset, optimizer, scheduler, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params
        self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
        self.step = 0
        self.is_master = True
        self.replica_id = replica_id

        beta = get_noise_schedule(self.params.noise_schedule)
 
        #beta = np.array(self.params.noise_schedule)
        noise_level = torch.cumprod(1 - beta, dim=0)**0.5
        noise_level = F.pad(noise_level, (1, 0), value=1.0)
        #noise_level = torch.cat([torch.tensor([1.0]), noise_level], dim=0)
        self.noise_level = noise_level.float()
        #noise_level = np.concatenate([[1.0], noise_level], axis=0)
        #self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.loss_fn = nn.L1Loss()
        self.summary_writer = None


    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
            'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
            'scheduler': self.scheduler.state_dict(),
            'params': dict(self.params),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict, load_scheduler=True):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def load_from_checkpoint(self, filename, load_scheduler=True):
        try:
            checkpoint = torch.load(filename)
            # if fine-tuning, don't load
            self.load_state_dict(checkpoint, load_scheduler=load_scheduler)
            return True
        except FileNotFoundError:
            print(f"{filename} not found")
            return False


    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def valid(self, dataset, name):
        device = next(self.model.parameters()).device
        loss_sum = 0.
        count = 0
        for batch in tqdm(dataset, desc=f'Validation {name}: Step {self.step}'):
            batch = _nested_map(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            audio = batch["audio"]
            if name == "a":
                features = batch["a_features"]
            elif name == 'v':
                features = batch["v_features"]
            else:
                features = batch["av_features"]

            if self.params.cond is None:
                loss = self.valid_step(features, audio)
            else:
                loss = self.valid_step(features, audio, batch["cond_labels"])

            loss_sum += loss.cpu().item()
            count += 1
        return loss_sum / count

    def train(self, max_steps=None):
        device = next(self.model.parameters()).device
        while True:
            for batch in tqdm(self.train_dataset, desc=f'Training: Epoch {self.step // len(self.train_dataset)}') if self.is_master else self.train_dataset:
                if max_steps is not None and self.step >= max_steps:
                    return
                batch = _nested_map(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

                # sample for a, v, av
                features = sample(batch["a_features"], batch["v_features"], batch["av_features"], sample_probs=self.params.sample_probs)

                if self.params.cond is None:
                    loss = self.train_step(features, batch["audio"])
                else:
                    loss = self.train_step(features, batch["audio"], batch["cond_labels"])

                #if torch.isnan(loss).any():
                #    raise runtimeerror(f'detected nan loss at step {self.step}.')
                if self.is_master:
                    if self.step != 0 and self.step % 5000 == 0 or self.step == max_steps - 1:
                        losses = {"train": loss}
                        for modal in ["a", "v", "av"]:
                            valid_loss = self.valid(dataset=self.valid_dataset, name=modal)
                            losses[f"valid_{modal}"] = valid_loss
                        self._write_summary(self.step, features, losses)
                    if self.step % 5000 == 0 or self.step == max_steps - 1:
                        self.save_to_checkpoint()
                self.step += 1

    def valid_step(self, features, audio, cond=None):
        #audio = features['audio']
        #spectrogram = features['spectrogram']

        N, T = audio.shape
        S = 1000
        device = audio.device
        self.noise_level = self.noise_level.to(device)

        with torch.no_grad():
            s = torch.randint(1, S + 1, [N], device=audio.device)
            l_a, l_b = self.noise_level[s-1], self.noise_level[s]
            noise_scale = l_a + torch.rand(N, device=audio.device) * (l_b - l_a)
            noise_scale = noise_scale.unsqueeze(1)
            noise = torch.randn_like(audio)
            noisy_audio = noise_scale * audio + (1.0 - noise_scale**2)**0.5 * noise

            predicted = self.model(noisy_audio, features, noise_scale.squeeze(1), cond=cond)
            loss = self.loss_fn(noise, predicted.squeeze(1))
        return loss

    def train_step(self, features, audio, cond=None):
        for param in self.model.parameters():
            param.grad = None

        #audio = features['audio']
        #spectrogram = features['spectrogram']

        N, T = audio.shape
        S = 1000
        device = audio.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            s = torch.randint(1, S + 1, [N], device=audio.device)
            l_a, l_b = self.noise_level[s-1], self.noise_level[s]
            noise_scale = l_a + torch.rand(N, device=audio.device) * (l_b - l_a)
            noise_scale = noise_scale.unsqueeze(1)
            noise = torch.randn_like(audio)
            noisy_audio = noise_scale * audio + (1.0 - noise_scale**2)**0.5 * noise

            predicted = self.model(noisy_audio, features, noise_scale.squeeze(1), cond=cond)
            loss = self.loss_fn(noise, predicted.squeeze(1))

        if not torch.isnan(loss).any():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            #print("lr", self.scheduler.get_last_lr()[0])
        return loss

    def _write_summary(self, step, features, losses):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        #writer.add_audio('audio/reference', features['audio'][0], step, sample_rate=self.params.sample_rate)
        for key, val in losses.items():
            writer.add_scalar(f"loss/{key}", val, step)
        #writer.add_scalar(f'train/train_loss', train_loss, step)
        #for name, valid_loss in valid_losses.items():
        #    writer.add_scalar(f'train/{name}_valid_loss', valid_loss, step)
        if hasattr(self, "grad_norm"):
            writer.add_scalar(f'train/grad_norm', self.grad_norm, step)
        writer.add_scalar(f'train/lr', self.scheduler.get_last_lr()[0], step)
        writer.flush()
        self.summary_writer = writer


def _train_impl(replica_id, model, train_dataset, valid_dataset, args, params):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    if not hasattr(params, "lr_scheduler") or params.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.max_steps, eta_min=params.min_lr)
    elif params.lr_scheduler == "cosine_w_warmup":
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=params.num_warmup_steps, num_training_steps=args.max_steps)
    elif params.lr_scheduler == "constant_w_warmup":
        scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=params.num_warmup_steps) 
    else:
        raise NotImplementedError(f"lr scheduler {params.lr_scheduler} not implemented.")

    learner = WaveGradLearner(replica_id, args.model_dir, model, train_dataset, valid_dataset, opt, scheduler, params, fp16=args.fp16)
    learner.is_master = (replica_id == 0)

    handler = SignalHandler(learner)
    signal.signal(signal.SIGUSR1, handler.handle)

    if args.ckpt is not None:
        # in fine-tuning case, don't load the scheduler
        learner.load_from_checkpoint(args.ckpt, load_scheduler=False)
        learner.scheduler.last_epoch = 0
        learner.step = 0

    restore_success = learner.restore_from_checkpoint()
    learner.train(max_steps=args.max_steps)


def train(args, params):
    wav_dir = os.path.join(args.train_root_dir, "data")
    feat_dir = os.path.join(args.train_root_dir, "features")
    train_data_loader = dataset_from_lists(args.train_wav_file, args.train_npy_files, params, wav_dir, feat_dir, cond_labels=args.train_cond, is_distributed=False)
    wav_dir = os.path.join(args.valid_root_dir, "data")
    feat_dir = os.path.join(args.valid_root_dir, "features")
    valid_data_loader = dataset_from_lists(args.valid_wav_file, args.valid_npy_files, params, wav_dir, feat_dir, cond_labels=args.valid_cond, is_valid=True, is_distributed=False)
    model = WaveGrad(params).cuda()
    _train_impl(0, model, train_data_loader, valid_data_loader, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = WaveGrad(params).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])

    wav_dir = os.path.join(args.train_root_dir, "data")
    feat_dir = os.path.join(args.train_root_dir, "features")
    train_data_loader = dataset_from_lists(args.train_wav_file, args.train_npy_files, params, wav_dir, feat_dir, cond_labels=args.train_cond, is_distributed=True)
    wav_dir = os.path.join(args.valid_root_dir, "data")
    feat_dir = os.path.join(args.valid_root_dir, "features")
    valid_data_loader = dataset_from_lists(args.valid_wav_file, args.valid_npy_files, params, wav_dir, feat_dir, cond_labels=args.valid_cond, is_distributed=False, is_valid=True)
    _train_impl(replica_id, model, train_data_loader, valid_data_loader, args, params)
