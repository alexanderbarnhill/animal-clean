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

    log.info("Initializing Model")
    model = AnimalClean(configuration)


    training_directory = get_training_directory(configuration)
    log.info(f"Putting relevant training output files in {training_directory}")

    log.info(f"Initializing Logger")
    logger = None
    if configuration.monitoring.method.lower() == "wandb":
        # wandb.login(key=configuration.secrets.monitoring.wandb.api_key)
        logger = pl_loggers.WandbLogger(project=configuration.monitoring_methods.wandb.project + f"-{configuration.name}", log_model="all",
                                        save_dir=training_directory)

        logger.watch(model, log="all")
    else:
        logger = pl_loggers.TensorBoardLogger(save_dir=training_directory)

    trainer = Trainer(
        accelerator="cpu" if not configuration.training.gpu else "gpu",
        log_every_n_steps=1,
        max_epochs=configuration.training.max_epochs,
        logger=logger,
        callbacks=get_callbacks(configuration),
        enable_checkpointing=configuration.training.enable_checkpointing,
        strategy="ddp",
        limit_train_batches=1.0 if not configuration.training.debugging.active else configuration.training.debugging.limit_train_batch
    )

    in_slurm = "SLURM_JOB_ID" in os.environ
    if in_slurm:
        log.info(f"Using Cluster Data Configuration")
    location = "local" if not in_slurm else "cluster"

    if use_human_speech_augmentation(configuration, loc=location):
        model.use_human_speech_augmentation = True
        model.human_speech_loader = get_human_speech_loader(configuration, loc=location)

    dataloaders = get_data_loaders(configuration, loc=location)

    log.info(f"Acquired data loaders")

    trainer.fit(model, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["val"])

    log.info(f"Training finished. Moving on to test phase")
    trainer.test(model=model, dataloaders=dataloaders["test"])

    if configuration.monitoring.method.lower() == "wandb":
        logger.experiment.unwatch(model)






