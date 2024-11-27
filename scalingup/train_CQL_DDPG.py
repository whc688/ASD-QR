# flake8: noqa
from copy import deepcopy
import logging
from typing import List, Optional, Tuple
import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger

import wandb
from scalingup.algo.algo import ScalingUpAlgo
from scalingup.inference import setup_inference
from scalingup.utils.generic import setup
import torch
import os
from scalingup.algo.CQL_DDPG_algo import CqlDdpgTrainingAlgo
from scalingup.utils.core import EnvSamplerConfig
import dataclasses

from pytorch_lightning.profilers import AdvancedProfiler


rank_idx = os.environ.get("NODE_RANK", 0)

torch.multiprocessing.set_sharing_strategy("file_system")


def setup_trainer(
    conf: OmegaConf,
    callbacks: Optional[List[Callback]] = None,
    wandb_logger: Optional[WandbLogger] = None,
) -> Tuple[LightningTrainer, ScalingUpAlgo]:
    if wandb_logger is None and rank_idx == 0:
        wandb_logger = setup(
            logdir=conf.logdir,
            seed=conf.seed,
            num_processes=conf.num_processes,
            tags=conf.tags,
            notes=conf.notes,
            conf=conf,
        )
    if conf.load_from_path is not None:
        algo = CqlDdpgTrainingAlgo.load_from_checkpoint(
            checkpoint_path=conf.load_from_path,
            **{
                k: (hydra.utils.instantiate(v) if "Config" in type(v).__name__ else v)
                for k, v in conf.algo.items()
                if not (k.endswith("_") and k.startswith("_"))
            },
            strict=False,
        )
        # TODO also load in ema model.
        # ideally always have it in checkpoint, without manual hacking
    else:
        algo = hydra.utils.instantiate(conf.algo)
    if callbacks is None:
        callbacks = []
    if rank_idx == 0:
        callbacks.extend(
            (
                ModelCheckpoint(
                    dirpath=f"{wandb.run.dir}/checkpoints/",  # type: ignore
                    filename="{epoch:04d}",
                    every_n_epochs=1,
                    save_last=True,
                    save_top_k=10,
                    monitor="epoch",
                    mode="max",
                    save_weights_only=False,
                ),
                RichProgressBar(leave=True),
                LearningRateMonitor(logging_interval="step"),
            )
        )

    profiler = AdvancedProfiler()

    trainer: LightningTrainer = hydra.utils.instantiate(
        conf.trainer,
        callbacks=callbacks,
        default_root_dir=wandb.run.dir if wandb.run is not None else None,  # type: ignore
        logger=wandb_logger,
        profiler=profiler,
    )
    return (
        trainer,
        algo,
    )


@hydra.main(config_path="config", config_name="train_CQL_DDPG", version_base="1.2")
def train(conf: OmegaConf):
    # only setup if rank is 0
    callbacks = []
    wandb_logger = None
    if conf.evaluation is not None and rank_idx == 0:
        policy, evaluation, wandb_logger = setup_inference(conf=conf)
        del evaluation
        del policy
    trainer, algo = setup_trainer(
        conf=conf, callbacks=callbacks, wandb_logger=wandb_logger
    )
    logging.info("Testing data loader")
    for _ in zip(range(5), algo.replay_buffer.get_loader(num_workers=1, batch_size=1)):
        pass

    trainer.fit(algo)
    wandb.finish()  # type: ignore
    print(trainer.profiler.describe())

if __name__ == "__main__":
    train()
