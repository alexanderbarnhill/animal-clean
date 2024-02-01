import math
import os

from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from data.audiodataset import DatabaseCsvSplit, get_audio_files_from_dir, Dataset, HumanSpeechBatchAugmentationDataset
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

def get_human_speech_opts(opts, loc="local"):
    if "human_speech" in opts.data:
        clean_dir = opts.data.human_speech.clean
        noisy_dir = opts.data.human_speech.noisy
        batch_percentage = opts.data.human_speech.batch_percentage
    else:
        if loc in opts.data:
            clean_dir = opts.data[loc].human_speech.clean
            noisy_dir = opts.data[loc].human_speech.noisy
            batch_percentage = opts.data[loc].human_speech.batch_percentage
        else:
            clean_dir = opts.data.human_speech.clean
            noisy_dir = opts.data.human_speech.noisy
            batch_percentage = opts.data.human_speech.batch_percentage
    if clean_dir is None or noisy_dir is None:
        return None, None, 0.0


    return clean_dir, noisy_dir, batch_percentage
def get_human_speech_loader(opts, loc="local"):
    clean_dir, noisy_dir, batch_percentage = get_human_speech_opts(opts, loc)

    log.info(f"Human speech augmentation during training is activated")
    log.info(f"Trying to organize clean and noisy samples for human speech augmentation")
    all_clean_files = glob(clean_dir + f"{os.sep}**{os.sep}*.wav", recursive=True)
    all_noisy_files = glob(noisy_dir + f"{os.sep}**{os.sep}*.wav", recursive=True)
    noisy_files_bases = [os.path.basename(f) for f in all_noisy_files]

    clean_files = []
    noisy_files = []
    for file in all_clean_files:
        if os.path.basename(file) in noisy_files_bases:
            idx = noisy_files_bases.index(os.path.basename(file))
            clean_files.append(file)
            noisy_files.append(all_noisy_files[idx])
    augmentation = opts.dataset.augmentation.active
    dataset = HumanSpeechBatchAugmentationDataset(
        split="train",
        file_names=clean_files,
        opts=opts,
        augmentation=augmentation,
        loc=loc,
        noise_directory=None,
    )
    dataset.noisy_files = noisy_files

    batch_size = int(math.floor(batch_percentage * opts.training.batch_size))
    log.info(f"Training Batch Size for Human Speech Samples: {batch_size}")
    loader = DataLoader(dataset=dataset,
                        shuffle=True,
                        batch_size=batch_size,
                        num_workers=opts.training.num_workers)
    return loader


def use_human_speech_augmentation(opts, loc="local"):
    clean, noisy, percentage = get_human_speech_opts(opts, loc)
    if clean is not None and percentage > 0.0:
        return True
    return False


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
        split_fracs, split_per_dir=True
    )

    audio_files = get_audio_files(input_data, data_dir)
    augmentation = opts.dataset.augmentation.active

    datasets = {
        split: Dataset(
            split=split,
            opts=opts,
            file_names=input_data.load(split, audio_files),
            augmentation=augmentation if split == "train" else False,
            noise_directory=opts.data[loc].noise_sources[split],
            loc=loc

        )
        for split in split_fracs.keys()
    }

    _, _, batch_percentage = get_human_speech_opts(opts, loc)
    if batch_percentage > 0.0:
        train_batch_size = opts.training.batch_size - (int(math.floor(batch_percentage * opts.training.batch_size)))
    else:
        train_batch_size = opts.training.batch_size

    log.info(f"Training Batch Size for Animal Samples: {train_batch_size}")
    batch_sizes = {
        "train": train_batch_size,
        "val": opts.training.batch_size,
        "test": opts.training.batch_size
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_sizes[split],
            shuffle=True if split == "train" else False,
            num_workers=opts.training.num_workers
        )
        for split in split_fracs.keys()
    }

    return dataloaders




