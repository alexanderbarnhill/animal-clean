import sys

from lightning import Trainer
from omegaconf import OmegaConf
import argparse
import os
from datetime import datetime
import logging as log
from lightning.pytorch import loggers as pl_loggers

from models.model import AnimalClean
from utilities.configuration import build_configuration
from utilities.training import *

parser = argparse.ArgumentParser()
parser.add_argument("--defaults", default=os.path.join(os.getcwd(), "configuration"))
parser.add_argument("--species_configuration", default=None)
parser.add_argument("--log_output", default=os.path.join(os.getcwd(), "logs"))

args = parser.parse_args()

if __name__ == '__main__':
    log_format = log.Formatter(fmt="%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = log.getLogger()
    rootLogger.setLevel(log.INFO)

    consoleHandler = log.StreamHandler()
    consoleHandler.setFormatter(log_format)
    consoleHandler.setLevel(log.INFO)
    consoleHandler.setStream(sys.stdout)

    os.makedirs(args.log_output, exist_ok=True)
    fileHandler = log.FileHandler(
        filename=os.path.join(args.log_output, f"train_{datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')}.log"),
        mode="w")
    fileHandler.setFormatter(log_format)
    fileHandler.setLevel(log.INFO)

    rootLogger.addHandler(consoleHandler)
    rootLogger.addHandler(fileHandler)

    log.info(f"Set File and Console Logging Handlers")
    log.info(f"Looking for configuration defaults in {args.defaults}")

    if args.species_configuration is not None:
        log.info(f"Adding species-specific configuration from {args.species_configuration}")

    configuration = build_configuration(args.defaults, args.species_configuration)
    in_slurm = "SLURM_JOB_ID" in os.environ

    if in_slurm:
        log.info(f"Using Cluster Data Configuration")
    location = "local" if not in_slurm else "cluster"

    run_name = get_run_name(configuration, location)
    log.info("Initializing Model")
    model = AnimalClean(configuration)

    training_directory = get_training_directory(configuration)
    log.info(f"Putting relevant training output files in {training_directory}")

    log.info(f"Initializing Logger")
    logger = None
    if configuration.monitoring.method.lower() == "wandb":
        # wandb.login(key=configuration.secrets.monitoring.wandb.api_key)
        logger = pl_loggers.WandbLogger(project=configuration.monitoring_methods.wandb.project + f"-{configuration.name}",
                                        log_model="all",
                                        save_dir=training_directory,
                                        name=run_name,
                                        id=run_name)

        logger.watch(model, log="all")
    else:
        logger = pl_loggers.TensorBoardLogger(save_dir=training_directory)

    train_batch_limitation = 1.0
    if configuration.training.debugging.active:
        train_batch_limitation = configuration.training.debugging.limit_train_batch
        log.info(f"Limiting batch size to {train_batch_limitation}")

    trainer = Trainer(
        accelerator="cpu" if not configuration.training.gpu else "gpu",
        log_every_n_steps=1,
        max_epochs=configuration.training.max_epochs,
        logger=logger,
        callbacks=get_callbacks(configuration),
        enable_checkpointing=configuration.training.enable_checkpointing,
        strategy="ddp",
        limit_train_batches=train_batch_limitation,
        accumulate_grad_batches=configuration.training.accumulate_gradient

    )

    if use_human_speech_augmentation(configuration, loc=location):
        model.use_human_speech_augmentation = True
        model.human_speech_loader = get_human_speech_loader(configuration, loc=location)

    dataloaders = get_data_loaders(configuration, loc=location)

    log.info(f"Acquired data loaders")

    ckpt_path = None

    if configuration.training.ckpt_path is not None and os.path.isfile(configuration.training.ckpt_path):
        log.info(f"Loading from checkpoint : {configuration.training.ckpt_path}")
        ckpt_path = configuration.training.ckpt_path
    elif configuration.training.ckpt_path is not None and not os.path.isfile(configuration.training.ckpt_path):
        log.info(f"Checkpoint path does not exist: {configuration.training.ckpt_path}")
        log.info(f"Cannot continue. Starting from scratch.")
    elif configuration.training.ckpt_path is not None and configuration.training.ckpt_path == "last":
        ckpt_path = "last"
        log.info(f"Attempting to resume training from last checkpoint")
    else:
        log.info(f"Starting training from scratch")

    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=ckpt_path
    )

    log.info(f"Training finished. Moving on to test phase")
    trainer.test(model=model, dataloaders=dataloaders["test"])

    if configuration.monitoring.method.lower() == "wandb":
        logger.experiment.unwatch(model)






