# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo11n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
from tqdm import tqdm

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)


class BaseTrainer:
    """
    A base class for creating trainers.

    This class provides the foundation for training YOLO models, handling the training loop, validation, checkpointing,
    and various training utilities. It supports both single-GPU and multi-GPU distributed training.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.

    Methods:
        train: Execute the training process.
        validate: Run validation on the test set.
        save_model: Save model training checkpoints.
        get_dataset: Get train and validation datasets.
        setup_model: Load, create, or download model.
        build_optimizer: Construct an optimizer for the model.

    Examples:
        Initialize a trainer and start training
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        # Update "-1" devices so post-training val does not repeat search
        self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
        self.validator = None
        self.metrics = None
        self.metrics_exit = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            YAML.save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo11n -> yolo11n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.data = self.get_dataset()

        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.fitness_exit = None
        self.loss = None
        self.ex_loss = None
        self.tloss = None
        self.ex_tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # 辅助去冻结哪些参数
        self.only_backbone = self.args.only_backbone

        # 置为路由器训练模式
        self.dynamicTrain = self.args.dynamicTrain

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            if not self.dynamicTrain:
                self._do_train(world_size)
            else:
                self._do_train_dy(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initialize and set the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names

        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer (by name) '{k}'")
                v.requires_grad = False
            # 仅训练backbone的时候，冻结exit分支
            elif self.only_backbone:
                # 按 '.' 分割参数名
                parts = k.split('.')
                if len(parts) >= 2 and parts[0] == 'model' and parts[1].isdigit():
                    layer_num = int(parts[1])  # 提取第2部分（索引1）作为层级数字
                    if layer_num >= 24:
                        LOGGER.info(f"Freezing layer (by index {layer_num}) '{k}'")
                        v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.metrics_exit = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _setup_train_dy(self, world_size):
        """Build dataloaders and optimizer on correct rank process for Router."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # 冻结除了路由器以外的所有参数
        for k, v in self.model.named_parameters():
            if 'router' not in k:
                print('freezing %s' % k)
                v.requires_grad = False

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=1,
                rank=-1,
                mode="val",
            )
            # TODO validator修改
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.metrics_exit = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer 超参直接在这里写死了
        hyp = {'lr0': 0.00001, 'weight_decay': 0.005, 'momentum': 0.937}
        nbs = 64  # nominal batch size
        self.accumulate = max(round(nbs / self.batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= self.batch_size * self.accumulate / nbs  # scale weight_decay

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if 'router' in k:
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d):
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

        self.optimizer = optim.AdamW(pg1, lr=hyp['lr0'], weight_decay=hyp['weight_decay'],
                                     betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum

        if len(pg0):
            self.optimizer.add_param_group({'params': pg0, 'weight_decay': 0})  # add pg0 without weight_decay
        if len(pg2):
            self.optimizer.add_param_group({'params': pg2, 'weight_decay': 0})  # add pg2 (biases)
        LOGGER.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train the model with the specified world size."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                if self.only_backbone:
                    LOGGER.info(self.progress_string())
                else:
                    LOGGER.info(self.ex_progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    raw_losses = self.model(batch)
                    loss, self.loss_items = raw_losses[0]
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                    # 分支的loss
                    if not self.only_backbone:
                        ex_loss, self.ex_loss_items = raw_losses[1]
                        self.ex_loss = ex_loss.sum()
                        self.ex_tloss = (
                            (self.ex_tloss * i + self.ex_loss_items) / (
                                        i + 1) if self.ex_tloss is not None else self.ex_loss_items
                        )

                # Backward
                if self.only_backbone:
                    self.scaler.scale(self.loss).backward()
                else:
                    self.scaler.scale(self.loss).backward(retain_graph=True)
                    self.scaler.scale(self.ex_loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if self.only_backbone:
                    if RANK in {-1, 0}:
                        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                                *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                                batch["cls"].shape[0],  # batch size, i.e. 8
                                batch["img"].shape[-1],  # imgsz, i.e 640
                            )
                        )
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)
                else:
                    if RANK in {-1, 0}:
                        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                        ex_loss_length = self.ex_tloss.shape[0] if len(self.ex_tloss.shape) else 1
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (2 + loss_length + ex_loss_length))
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                                *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                                *(self.ex_tloss if ex_loss_length > 1 else torch.unsqueeze(self.ex_tloss, 0)),
                                batch["cls"].shape[0],  # batch size, i.e. 8
                                batch["img"].shape[-1],  # imgsz, i.e 640
                            )
                        )
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            # end of batch

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)  # prevent VRAM spike
                    self.metrics, self.fitness = self.validate(only_backbone=True)
                # Branchy Validation 验证分支的结果
                if self.args.val and not self.only_backbone:
                    self._clear_memory(threshold=0.5)
                    self.metrics_exit, self.fitness_exit = self.validate(only_backbone=False)
                    # 修改 self.metrics_exit 的键的名称 为所有键名添加"EX"前缀 方便之后记录
                    self.metrics_exit = {f"EX{key}": value for key, value in self.metrics_exit.items()}
                    # TODO 把self.metrics_exit拼接到self.metrics当中？
                # 给两个出口的综合得分标准赋予不同权重，倾向于选择分支效果更好的模型
                # TODO 将超参提取出来 self.metrics保存模型呢块不改了
                self.fitness = 0.3 * self.fitness + 0.7 * self.fitness_exit
                # 由于需要计算两个fitness 所以best_fitness的标准也要发生变化
                if not self.best_fitness or self.best_fitness < self.fitness:
                    self.best_fitness = self.fitness

                self.save_metrics(metrics={
                    **self.label_loss_items(self.tloss),
                    **self.metrics,
                    **self.metrics_exit,
                    **self.lr})

                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1
        # end of epoch

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def _do_train_dy(self, world_size=1):
        """Train the Router."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train_dy(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        # TODO 修改epoch退出条件
        while True:
            mloss = torch.zeros(4, device=self.device)  # mean losses
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(('\n' + '%10s' * 8) % (
                'Epoch', 'gpu_mem', 'diff', 'score.0', 'score.1', 'total', 'labels', 'img_size'))
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize
                if ni % self.accumulate == 0:
                    self.scaler.step(self.optimizer)  # optimizer.step
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update(self.model)

                # Log
                if RANK in {-1, 0}:
                    mloss = (mloss * i + self.loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 + '%10.4g' * 6) % (
                        '%g/%g' % (epoch, self.epochs - 1),
                        mem,
                        *mloss,
                        batch["cls"].shape[0],
                        batch["img"].shape[-1])
                    pbar.set_description(s)
                    self.run_callbacks("on_batch_end")
                    # if self.args.plots and ni in self.plot_idx:
                    #     self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
            # end of batch ---------------------------------------------------------------------------------------------

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)  # prevent VRAM spike
                    self.metrics, self.fitness = self.validate(only_backbone=True, dynamic=True)
                    # print("观察如何统计self.loss_items {}".format(val_loss_items))
                # 仅凭借val_loss作为选取最优模型的标准 TODO 思考更好的标准
                if not self.best_fitness or self.best_fitness < self.fitness:
                    self.best_fitness = self.fitness

                self.save_metrics(metrics={
                    **self.label_loss_items_dy(mloss),
                    **self.metrics,
                    **self.lr})

                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1
        # end of epoch -------------------------------------------------------------------------------------------------

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            # self.final_eval()
            # if self.args.plots:
            #     self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def _get_thres(self, batch_size=32, imgsz=640, augment=False, half_precision=True):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        # Dataloader
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.data = self.get_dataset()
        self.val_loader = self.get_dataloader(
            self.data["val"], batch_size=batch_size, rank=LOCAL_RANK, mode="val"
        )

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.float().fuse().eval()

        # Compatibility updates
        for m in self.model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training
        # TODO 什么情况下model不止是有模块？
        self.model.get_score = True

        # Half
        half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
        if half:
            self.model.half()

        # TODO 为什么此处半精度有问题？
        self.model.float()
        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.model.parameters())))  # run once

        print(self.val_loader)
        score_list = []
        for batch_i, img_batch in enumerate(tqdm(self.val_loader)):
            img = img_batch["img"]
            img = img.to(device, non_blocking=True)
            # img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img.float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            with torch.no_grad():
                # Run model
                cur_score = self.model(img, augment=augment)  # inference
                for item in cur_score:
                    score_list.append(item.item())

        thres = ['0']
        for i in list(range(10, 100, 10)):
            thres.append(str(np.percentile(score_list, i)))
        thres.append('1')
        return thres

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2 ** 30)

    def _clear_memory(self, threshold: float = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dictionary using pandas."""
        import pandas as pd  # scope for faster 'import ultralytics'

        return pd.read_csv(self.csv).to_dict(orient="list")

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        if not self.dynamicTrain:
            for n, m in self.model.named_modules():
                if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint

    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """
        Load, create, or download model for any task.

        Returns:
            (dict): Optional checkpoint to resume training from.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allow custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self, only_backbone=True, dynamic=False):
        """
        Run validation on test set using self.validator.

        Args:
            dynamic (bool): 表示此时是在训练路由器
            only_backbone (bool): 作为区别验证模式的标志，验证main分支还是exit分支.

        Returns:
            metrics (dict): Dictionary of validation metrics.
            fitness (float): Fitness score for the validation.
        """
        metrics = self.validator(self, only_backbone=only_backbone, dynamic=dynamic)
        if not dynamic:
            fitness = metrics.pop("fitness",
                                  -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        else:
            fitness = metrics.pop("val/adaptive_loss")
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Return a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Return dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def label_loss_items_dy(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        loss_names = ["loss_diff", "score", "1-score", "adaptive_loss"]
        keys = [f"{prefix}/{x}" for x in loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Return a string describing training progress."""
        return ""

    def ex_progress_string(self):
        """联合训练时的日志需要查看exit分支的训练情况."""
        loss_len = 3
        return ("\n" + "%11s" * (4 + loss_len * 2)) % (
            "Epoch",
            "GPU_mem",
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "ex_box_loss",
            "ex_cls_loss",
            "ex_dfl_loss",
            "Instances",
            "Size",
        )

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plot training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plot training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        # TODO 添加训练exit分支的记录
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # header
        t = time.time() - self.train_time_start
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Perform final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                        "imgsz",
                        "batch",
                        "device",
                        "close_mosaic",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer
