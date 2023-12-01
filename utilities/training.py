import os

from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from data.audiodataset import DatabaseCsvSplit, get_audio_files_from_dir, Dataset
import logging as log
import torch
from glob import glob

def get_training_directory(opts):
    if opts.training.training_directory is None:
        path = os.path.join(os.path.join(os.path.dirname(__file__), ".."), "training_output")
    else:
        path = opts.training.training_directory
    os.makedirs(path, exist_ok=True)
    return path


def get_callbacks(opts):
    callbacks = [
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        EarlyStopping(monitor=opts.training.early_stopping.monitor,
                      mode=opts.training.early_stopping.mode,
                      patience=opts.training.early_stopping.patience
                      )
    ]

    if opts.training.enable_checkpointing:
        callbacks.append(ModelCheckpoint(monitor="val_loss", save_top_k=5))
    return callbacks


def get_audio_files(input_data, data_dir):
    audio_files = None
    if input_data.can_load_from_csv():
        log.info("Found csv files in {}".format(data_dir))
    else:
        log.debug("Searching for audio files in {}".format(data_dir))
        audio_files = list(get_audio_files_from_dir(data_dir))
        log.info("Found {} audio files for training.".format(len(audio_files)))
        if len(audio_files) == 0:
            log.error(f"No audio files found")
            exit(1)
    return audio_files


def get_data_loaders(opts, loc="local"):
    log.info(f"Location Information: {loc}")
    if "data_directory" in opts.data:
        data_dir = opts.data.data_directory
    else:
        if loc in opts.data:
            data_dir = opts.data[loc].data_directory
        else:
            data_dir = opts.data.data_directory
    if data_dir is None:
        raise ValueError(f"Data directory could not be found!")
    log.info(f"Looking for data in {data_dir}")
    split_fracs = {"train": .7, "val": .15, "test": .15}
    input_data = DatabaseCsvSplit(
        split_fracs, working_dir=data_dir, split_per_dir=True
    )

    audio_files = get_audio_files(input_data, data_dir)
    augmentation = opts.dataset.augmentation.active

    datasets = {
        split: Dataset(
            split=split,
            opts=opts,
            file_names=input_data.load(split, audio_files),
            augmentation=augmentation if split == "train" else False,
            noise_files=glob(opts.data[loc].noise_sources[split] + f"{os.sep}**/*{os.sep}.wav", recursive=True),
            loc=loc

        )
        for split in split_fracs.keys()
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=opts.training.batch_size,
            shuffle=True if split == "train" else False,
            num_workers=opts.training.num_workers
        )
        for split in split_fracs.keys()
    }

    return dataloaders




