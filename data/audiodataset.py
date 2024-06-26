"""
Module: audiodataset.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import sys
import csv
import glob
import random
import pathlib
import numpy as np
import soundfile as sf
import data.transforms as T

import torch
import torch.utils.data
import torch.multiprocessing as mp

import data.signal as signal

from math import ceil
from skimage import exposure
from types import GeneratorType
import logging as log
from collections import defaultdict
from utilities.FileIO import AsyncFileReader
import json
from typing import Any, Dict, Iterable, List
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Data preprocessing default options
"""
DefaultSpecDatasetOps = {
    "sr": 44100,
    "preemphases": 0.98,
    "n_fft": 4096,
    "hop_length": 441,
    "n_freq_bins": 256,
    "fmin": 500,
    "fmax": 10000,
    "freq_compression": "linear",
    "min_level_db": -100,
    "ref_level_db": 20,
}

"""
Get audio files from directory
"""


def get_audio_files_from_dir(path: str):
    if path is None or len(path) == 0:
        return []
    audio_files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
    audio_files = map(lambda p: pathlib.Path(p), audio_files)
    audio_files = filter(lambda p: not p.match("*.bkp/*"), audio_files)
    base = pathlib.Path(path)
    return list(map(lambda p: str(p.relative_to(base)), audio_files))


"""
Helper class in order to speed up filtering potential broken files
"""


class _FilterPickleHelper(object):
    def __init__(self, predicate, *pred_args):
        self.predicate = predicate
        self.args = pred_args

    def __call__(self, item):
        return self.predicate(item, *self.args)


"""
Parallel Filtering to analyze incoming data files
"""


class _ParallelFilter(object):
    def __init__(self, iteratable, n_threads=None, chunk_size=1):
        self.data = iteratable
        self.n_threads = n_threads
        self.chunk_size = chunk_size

    def __call__(self, func, *func_args):
        with mp.Pool(self.n_threads) as pool:
            func_pickle = _FilterPickleHelper(func, *func_args)
            for keep, c in pool.imap_unordered(func_pickle, self.data, self.chunk_size):
                if keep:
                    yield c


"""
Analyzing loudness criteria of each audio file by checking maximum amplitude (default: 1e-3)
"""


def _loudness_criteria(file_name: str, working_dir: str = None):
    if working_dir is not None:
        file_path = os.path.join(working_dir, file_name)
    else:
        file_path = file_name
    y, __ = sf.read(file_path, always_2d=True, dtype="float32")
    max_ampl = y.max()
    if max_ampl < 1e-3:
        return True, file_name
    else:
        return False, None


"""
Filtering all audio files in previous which do not fulfill the loudness criteria
"""


def get_broken_audio_files(files: Iterable[str], working_dir: str = None):
    f = _ParallelFilter(files, chunk_size=100)
    return f(_loudness_criteria, working_dir)


"""
Computes the CSV Split in order to prepare for randomly partitioning all data files into a training, validation, and test corpus
by dividing the data in such a way that audio files of a given tape are stored only in one of the three partitions.
The filenames per dataset will be stored in CSV files (train.csv, val.csv, test.csv). Each CSV File will be merged into
a train, val, and test file holding the information how a single partition is made up from single CSV files. These three
files reflects the training, validation, and test set.
"""


class CsvSplit(object):

    def __init__(
            self,
            split_fracs: Dict[str, float],
            working_dir: (str) = None,
            seed: (int) = None,
            split_per_dir=False,
    ):
        if not np.isclose(np.sum([p for _, p in split_fracs.items()]), 1.):
            raise ValueError("Split probabilities have to sum up to 1.")
        self.split_fracs = split_fracs
        self.working_dir = working_dir
        self.seed = seed
        self.split_per_dir = split_per_dir
        self.splits = defaultdict(list)

    """
    Return split for given partition. If there is already an existing CSV split return this split if it is valid or
    in case there exist not a split yet generate a new CSV split
    """

    def load(self, split: str, files: List[Any] = None):
        if split not in self.split_fracs:
            raise ValueError(
                "Provided split '{}' is not in `self.split_fracs`.".format(split)
            )

        if self.splits[split]:
            return self.splits[split]
        if self.working_dir is None:
            self.splits = self._split_with_seed(files)
            return self.splits[split]
        if self.can_load_from_csv():
            log.info(f"Loading from CSV")
            if not self.split_per_dir:
                csv_split_files = {
                    split_: (os.path.join(self.working_dir, split_ + ".csv"),)
                    for split_ in self.split_fracs.keys()
                }
            else:
                csv_split_files = {}
                for split_ in self.split_fracs.keys():
                    split_file = os.path.join(self.working_dir, split_)
                    csv_split_files[split_] = []
                    with open(split_file, "r") as f:
                        for line in f.readlines():
                            csv_split_files[split_].append(line.strip())

            for split_ in self.split_fracs.keys():
                for csv_file in csv_split_files[split_]:
                    if not csv_file or csv_file.startswith(r"#"):
                        continue
                    csv_file_path = os.path.join(self.working_dir, csv_file)
                    log.info(f"Looking in {csv_file_path}")
                    with open(csv_file_path, "r") as f:
                        reader = csv.reader(f)
                        for item in reader:
                            if len(item) != 1:
                                continue
                            file_ = os.path.basename(item[0])
                            file_ = os.path.join(os.path.dirname(csv_file), file_)
                            self.splits[split_].append(file_)
            return self.splits[split]

        if not self.split_per_dir:
            working_dirs = (self.working_dir,)
        else:
            f_d_map = self._get_f_d_map(files)
            working_dirs = [os.path.join(self.working_dir, p) for p in f_d_map.keys()]
        for working_dir in working_dirs:
            splits = self._split_with_seed(
                files if not self.split_per_dir else f_d_map[working_dir]
            )
            for split_ in splits.keys():
                csv_file = os.path.join(working_dir, split_ + ".csv")
                log.debug("Generating {}".format(csv_file))
                if self.split_per_dir:
                    with open(os.path.join(self.working_dir, split_), "a") as f:
                        p = pathlib.Path(csv_file).relative_to(self.working_dir)
                        f.write(str(p) + "\n")
                if len(splits[split_]) == 0:
                    raise ValueError(
                        "Error splitting dataset. Split '{}' has 0 entries".format(
                            split_
                        )
                    )
                with open(csv_file, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    for item in splits[split_]:
                        writer.writerow([item])
                self.splits[split_].extend(splits[split_])
        return self.splits[split]

    """
    Check whether it is possible to correctly load information from existing csv files
    """

    def can_load_from_csv(self):
        if not self.working_dir:
            return False
        if self.split_per_dir:
            for split in self.split_fracs.keys():
                split_file = os.path.join(self.working_dir, split)
                if not os.path.isfile(split_file):
                    return False
                log.debug("Found dataset split file {}".format(split_file))
                with open(split_file, "r") as f:
                    for line in f.readlines():
                        csv_file = line.strip()
                        if not csv_file or csv_file.startswith(r"#"):
                            continue
                        if not os.path.isfile(os.path.join(self.working_dir, csv_file)):
                            log.error("File not found: {}".format(csv_file))
                            raise ValueError(
                                "Split file found, but csv files are missing. "
                                "Aborting..."
                            )
        else:
            for split in self.split_fracs.keys():
                csv_file = os.path.join(self.working_dir, split + ".csv")
                if not os.path.isfile(csv_file):
                    return False
                log.debug("Found csv file {}".format(csv_file))
        return True

    """
    Create a mapping from directory to containing files.
    """

    def _get_f_d_map(self, files: List[Any]):

        f_d_map = defaultdict(list)
        if self.working_dir is not None:
            for f in files:
                f_d_map[str(pathlib.Path(self.working_dir).joinpath(f).parent)].append(
                    f
                )
        else:
            for f in files:
                f_d_map[str(pathlib.Path(".").resolve().joinpath(f).parent)].append(f)
        return f_d_map

    """
    Randomly splits the dataset using given seed
    """

    def _split_with_seed(self, files: List[Any]):
        if not files:
            raise ValueError("Provided list `files` is `None`.")
        if self.seed:
            random.seed(self.seed)
        return self.split_fn(files)

    """
    A generator function that returns all values for the given `split`.
    """

    def split_fn(self, files: List[Any]):
        _splits = np.split(
            ary=random.sample(files, len(files)),
            indices_or_sections=[
                int(p * len(files)) for _, p in self.split_fracs.items()
            ],
        )
        splits = dict()
        for i, key in enumerate(self.splits.keys()):
            splits[key] = _splits[i]
        return splits


"""
Extracts the year and tape from the given audio filename (filename structure: call-label_ID_YEAR_TAPE_STARTTIME_ENDTIME)
"""


def get_tape_key(file, valid_years=None):
    while "__" in file:
        file = file.replace("__", "_")
    try:
        attributes = file.split(sep="_")
        year = attributes[-4]
        tape = attributes[-3]
        if valid_years is not None and int(year) not in valid_years:
            return None
        return year + "_" + tape.upper()
    except Exception:
        import traceback
        print("Warning: skippfing file {}\n{}".format(file, traceback.format_exc()))
        pass
    return None


"""
Splits a given list of file names across different partitions.
"""


class DatabaseCsvSplit(CsvSplit):
    valid_years = set(range(1950, 2200))

    """
    Count the samples per tape.
    """

    def split_fn(self, files: Iterable[Any]):
        if isinstance(files, GeneratorType):
            files = list(files)
        n_files = len(files)
        tapes = defaultdict(int)
        for file in files:
            try:
                key = get_tape_key(file, self.valid_years)
                if key is not None:
                    tapes[key] += 1
                else:
                    n_files -= 1
            except IndexError:
                n_files -= 1
                pass

        tape_names = list(tapes)

        """
        Helper class which creates a mapping (per fraction) in order to handle added tapes and number of files per tape
        """

        class Mapping:
            def __init__(self):
                self.count = 0
                self.names = []

            def add(self, name, count):
                self.count += count
                self.names.append(name)

        mappings = {s: Mapping() for s in self.split_fracs.keys()}

        for tape_name in tape_names:
            missing_files = {
                s: n_files * f - mappings[s].count for s, f in self.split_fracs.items()
            }
            r = random.uniform(0., sum(f for f in missing_files.values()))
            for _split, _n_files in missing_files.items():
                r -= _n_files
                if r < 0:
                    mappings[_split].add(tape_name, tapes[tape_name])
                    break
            assert r < 0, "Should not get here"

        splits = defaultdict(list)
        for file in files:
            tape = get_tape_key(file, self.valid_years)
            if tape is not None:
                for s, m in mappings.items():
                    if tape in m.names:
                        splits[s].append(file)

        return splits


"""
Dataset for that returns just the provided file names.
"""


class FileNameDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            file_names: Iterable[str],
            working_dir=None,
            transform=None,
            logger_name="TRAIN",
            dataset_name=None,
    ):
        if isinstance(file_names, GeneratorType):
            self.file_names = list(file_names)
        else:
            self.file_names = file_names
        self.working_dir = working_dir
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        if not isinstance(self.file_names, list):
            self.file_names = list(self.file_names)
        return len(self.file_names)

    def __getitem__(self, idx):
        if self.working_dir:
            return os.path.join(self.working_dir, self.file_names[idx])
        sample = self.file_names[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


"""
Dataset for loading audio data.
"""


class AudioDataset(FileNameDataset):

    def __init__(
            self,
            file_names: Iterable[str],
            working_dir=None,
            sr=44100,
            mono=True,
            *args,
            **kwargs
    ):
        super().__init__(file_names, working_dir, *args, **kwargs)

        self.sr = sr
        self.mono = mono

    def __getitem__(self, idx):
        file = self.file_names[idx]
        if self.working_dir is not None:
            file = os.path.join(self.working_dir, file)
        sample = T.load_audio_file(file, self.sr, self.mono)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


"""
Dataset to load audio files of each partition and compute several data preprocessing steps (resampling, augmentation, compression, subsampling/padding, etc.).
Filenames have to fulfill the follwing structure in order to ensure a correct data processing:  --- call/noise-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME ---

Single components of the filename template:
--------------------------------------------
1. LABEL = a placeholder for any kind of string which describes the label of the respective sample, e.g. call-N9, orca, echolocation, etc.
2. ID = unique ID to identify the audio clip
3. YEAR = year when the tape was recorded
4. TAPENAME = name of the recorded tape (has to be unique in order to do a proper data split into train, devel, test set by putting one tape only in one of the three sets
5. STARTTIME = start time of the clip in milliseconds (integer number, e.g 123456ms = 123.456s
5. ENDTIME = end time of the clip in milliseconds
"""


class Dataset(AudioDataset):
    """
    Create variables in order to filter the filenames whether it is a target signal (call) or a noise signal (noise). Moreover
    the entire spectral transform pipeline is created in oder to set up the data preprocessing for each audio file.
    """

    def __init__(
            self,
            split,
            opts,
            augmentation,
            file_names: Iterable[str],
            noise_directory: str or None,
            loc=None,
            m=None,
            *args,
            **kwargs
    ):
        self.m = m
        self.logfix = f"[Dataset -- {split}]:"
        self.mode = split
        log.info(f"{self.logfix} Using {len(file_names)} files")
        if noise_directory is not None:
            log.info(f"{self.logfix} Looking for noise files in {noise_directory}")
            noise_files = glob.glob(noise_directory + f"{os.sep}**{os.sep}*.wav", recursive=True)
        else:
            log.info(f"{self.logfix} No noise directory defined. Not using real world noise")
            noise_files = None
        if noise_files is not None:
            log.info(f"{self.logfix} {len(noise_files)} noise files")
        else:
            log.info(f"{self.logfix} No noise files found.")
        self.dataset_opts = opts.dataset
        self.data_opts = opts.data
        if loc is None or loc not in self.data_opts:
            dir_opts = self.data_opts
        else:
            dir_opts = self.data_opts[loc]
        working_dir = dir_opts.working_directory
        if working_dir is None:
            working_dir = dir_opts.data_directory
        self.cache_dir = dir_opts.cache_directory
        self.sr = self.dataset_opts.sample_rate
        super().__init__(file_names, working_dir, self.sr, *args, **kwargs)
        if self.dataset_name is not None:
            log.info(f"{self.logfix} Initializing dataset {self.dataset_name}...")

        self.sp = signal.signal_proc()

        self.aug_opts = self.dataset_opts.augmentation
        self.sample_rate = self.dataset_opts.sample_rate
        self.df = self.aug_opts.noise_masking.df
        self.exp_e = self.aug_opts.noise_masking.exp_e
        self.bin_pow = self.aug_opts.noise_masking.bin_pow
        self.gaus_mean = self.aug_opts.noise_masking.gauss_mean
        self.gaus_stdv = self.aug_opts.noise_masking.gauss_std
        self.poisson_lambda = self.aug_opts.noise_masking.poisson_lambda
        self.orig_noise_value = self.aug_opts.noise_masking.original_noise_value
        self.masking_probability = self.aug_opts.noise_masking.masking_probability
        self.do_masking = False
        self.foreign_masking = False
        if self.masking_probability == 0.0:
            log.info(f"{self.logfix} Creation of Binary Masks disabled")
        elif self.masking_probability == 1.0:
            log.info(f"{self.logfix} Additive noise masking disabled -- only using binary masks")
            self.do_masking = True
        else:
            log.info(f"{self.logfix} Binary masks will be created with probability {self.masking_probability}")
            self.do_masking = True

        masking_data = self.aug_opts.noise_masking.masking_data
        # SETUP MASKING DATA
        if masking_data is not None and masking_data != "target" and os.path.isdir(masking_data):
            log.info(f"Using alternative data for mask creation")
            log.info(f"Mask creation data path: {masking_data}")
            self.masking_data = glob.glob(masking_data + "/**/**/*.wav", recursive=True)
            self.foreign_masking = True
        else:
            log.info(f"Using target data for mask creation")
            self.masking_data = self.file_names

        self.do_masking = ((self.mode == "train" and self.foreign_masking and self.do_masking) or
                           (not self.foreign_masking and self.do_masking))

        if self.do_masking:
            log.info(f"Number of files used for masking generation: {len(self.masking_data)}")

        self.freq_compression = self.dataset_opts.frequency.compression
        self.f_min = self.dataset_opts.frequency.frequency_min
        self.f_max = self.dataset_opts.frequency.frequency_max
        self.n_fft = self.dataset_opts.fft.n_fft
        self.random = random
        self.seq_len = self.dataset_opts.feature_size.sequence_length
        self.n_freq_bins = self.dataset_opts.feature_size.n_freq_bins
        self.hop_length = self.dataset_opts.fft.hop
        self.augmentation = augmentation
        self.file_reader = AsyncFileReader()
        self.noise_files = noise_files
        self.min_thres_detect = self.dataset_opts.signal_detection.min_thres_detect
        self.max_thres_detect = self.dataset_opts.signal_detection.max_thres_detect
        self.perc_of_max_signal = self.dataset_opts.signal_detection.perc_of_max_signal

        valid_freq_compressions = ["linear", "mel", "mfcc"]

        if self.freq_compression not in valid_freq_compressions:
            raise ValueError(
                "{} is not a valid freq_compression. Must be one of {}",
                format(self.freq_compression, valid_freq_compressions),
            )

        log.debug(
            f"{self.logfix} Number of files to denoise : {len(self.file_names)}"
        )
        self.transform_opts = {
            "target": {
                "sr": self.sample_rate,
                "pre_emphasis": self.dataset_opts.pre_emphasis,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "amplitude_active": self.aug_opts.amplitude_shift.active,
                "increase_db": self.aug_opts.amplitude_shift.increase_db,
                "decrease_db": self.aug_opts.amplitude_shift.decrease_db,
                "time_active": self.aug_opts.time_stretch.active,
                "time_from": self.aug_opts.time_stretch.from_,
                "time_to": self.aug_opts.time_stretch.to_,
                "pitch_active": self.aug_opts.pitch_shift.active,
                "pitch_from": self.aug_opts.pitch_shift.from_,
                "pitch_to": self.aug_opts.pitch_shift.to_,
                "max_snr": self.aug_opts.noise_addition.max_snr,
                "min_snr": self.aug_opts.noise_addition.min_snr,
                "min_level_db": self.dataset_opts.min_level_db,
                "ref_level_db": self.dataset_opts.ref_level_db,
                "normalize": self.dataset_opts.normalization.method,
                "f_min": self.f_min,
                "f_max": self.f_max
            },
            ### USE ORCA SETTINGS TODO MOVE THIS
            "foreign": {
                "sr": 44100,
                "pre_emphasis": 0.98,
                "n_fft": 4096,
                "hop_length": 441,
                "amplitude_active": True,
                "increase_db": 3,
                "decrease_db": -6,
                "time_active": True,
                "time_from": 0.5,
                "time_to": 2,
                "pitch_active": True,
                "pitch_from": 0.5,
                "pitch_to": 1.5,
                "max_snr": -2,
                "min_snr": -8,
                "min_level_db": -100,
                "ref_level_db": 20,
                "normalize": "db",
                "f_min": 500,
                "f_max": 10000
            }
        }

        self.set_transforms(source="target")
        log.info(f"Target Settings:")
        log.info(f"{json.dumps(self.transform_opts['target'], indent=4)}")
        log.info(f"Foreign Masking Settings:")
        log.info(f"{json.dumps(self.transform_opts['foreign'], indent=4)}")

    def set_transforms(self, source="target"):
        transform_opts = self.transform_opts[source]
        spec_transforms = [
            lambda fn: T.load_audio_file(fn, sr=transform_opts["sr"]),
            T.PreEmphasize(transform_opts["pre_emphasis"]),
            T.Spectrogram(transform_opts["n_fft"], transform_opts["hop_length"], center=False),
        ]

        if self.cache_dir is None:
            self.t_spectrogram = T.Compose(spec_transforms)
        else:
            self.t_spectrogram = T.CachedSpectrogram(
                cache_dir=self.cache_dir,
                spec_transform=T.Compose(spec_transforms),
                n_fft=transform_opts["n_fft"],
                hop_length=transform_opts["hop_length"],
                file_reader=self.file_reader)

        if self.augmentation:

            log.debug(f"{self.logfix} Init augmentation transforms")
            if transform_opts["amplitude_active"]:

                self.t_amplitude = T.RandomAmplitude(
                    increase_db=transform_opts["increase_db"],
                    decrease_db=transform_opts["decrease_db"])
            else:
                self.t_amplitude = T.RandomAmplitude(
                    increase_db=1,
                    decrease_db=1)
            if transform_opts["time_active"]:
                self.t_timestretch = T.RandomTimeStretch(
                    from_=transform_opts["time_from"],
                    to_=transform_opts["time_to"]
                )
            else:
                self.t_timestretch = T.RandomTimeStretch(
                    from_=1,
                    to_=1
                )
            if transform_opts["pitch_active"]:

                self.t_pitchshift = T.RandomPitchSift(
                    from_=transform_opts["pitch_from"],
                    to_=transform_opts["pitch_to"]
                )
            else:
                self.t_pitchshift = T.RandomPitchSift(
                    from_=1,
                    to_=1
                )
        else:
            # only for noise augmentation during validation phase - intensity, time and pitch augmentation is not used during validation/test
            if transform_opts["time_active"]:
                self.t_timestretch = T.RandomTimeStretch(
                    from_=transform_opts["time_from"],
                    to_=transform_opts["time_to"]
                )
            else:
                self.t_timestretch = T.RandomTimeStretch(
                    from_=1,
                    to_=1
                )

            if transform_opts["pitch_active"]:

                self.t_pitchshift = T.RandomPitchSift(
                    from_=transform_opts["pitch_from"],
                    to_=transform_opts["pitch_to"]
                )
            else:
                self.t_pitchshift = T.RandomPitchSift(
                    from_=1,
                    to_=1
                )
            log.debug(f"{self.logfix} Running without intensity, time, and pitch augmentation")

        if self.freq_compression == "linear":
            self.t_compr_f = T.Interpolate(256, transform_opts["sr"], transform_opts["f_min"], transform_opts["f_max"])
        elif self.freq_compression == "mel":
            self.t_compr_f = T.F2M(sr=transform_opts["sr"], n_mels=256, f_min=transform_opts["f_min"],
                                   f_max=transform_opts["f_max"])
        elif self.freq_compression == "mfcc":
            self.t_compr_f = T.Compose(T.F2M(sr=transform_opts["sr"], n_mels=256, f_min=transform_opts["f_min"],
                                             f_max=transform_opts["f_max"]))
            self.t_compr_mfcc = T.M2MFCC(n_mfcc=32)
        else:
            raise "Undefined frequency compression"

        if self.noise_files is not None and len(self.noise_files) > 0:
            log.debug(f"{self.logfix} Init training real-world noise files for noise2noise adding")
            self.t_addnoise = T.RandomAddNoise(
                self.noise_files,
                self.t_spectrogram,
                T.Compose(self.t_timestretch, self.t_pitchshift, self.t_compr_f),
                min_length=self.seq_len,
                min_snr=transform_opts["min_snr"],
                max_snr=transform_opts["max_snr"],
                return_original=True
            )
        else:
            self.t_addnoise = None

        self.t_compr_a = T.Amp2Db(min_level_db=transform_opts["min_level_db"])

        min_max_normalize = transform_opts["normalize"] == "min_max"
        if min_max_normalize:
            self.t_norm = T.MinMaxNormalize()
            log.debug(f"{self.logfix} Init min-max-normalization activated")
        else:
            self.t_norm = T.Normalize(
                min_level_db=transform_opts["min_level_db"],
                ref_level_db=transform_opts["ref_level_db"],
            )
            log.debug(f"{self.logfix} Init 0/1-dB-normalization activated")

        self.t_subseq = T.PaddedSubsequenceSampler(self.seq_len, dim=1, random=self.augmentation)

    def __getitem__(self, idx):
        """
            Computes per filename the entire data preprocessing pipeline containing all transformations and returns the
            preprocessed sample as well as the ground truth label
        """
        self.clone = False
        self.orig_noise = False
        self.binary_orig = False
        self.binary_mask = False
        self.binary_ones_pow = False
        self.binary_orig_pow = False

        min_dist = 0 if self.noise_files is not None and len(
            self.noise_files) > 0 and self.t_addnoise is not None else 1

        noise_method = random.random()
        if noise_method < self.masking_probability and self.do_masking:
            # Do Masking Only
            min_dist = 6
            max_dist = 9
        else:
            max_dist = 5

        distribution_idx = random.randint(min_dist, max_dist)
        file_name = self.file_names[idx]

        foreign_sample = False
        transform_opts = self.transform_opts["target"]
        if 6 <= distribution_idx <= 9 and self.foreign_masking:
            idx = random.randint(0, len(self.masking_data) - 1)
            file_name = self.masking_data[idx]
            ### SET TRANSFORMS TO ORCA ###
            self.set_transforms("foreign")
            foreign_sample = True
            transform_opts = self.transform_opts["foreign"]

        if not foreign_sample:
            ### SET TRANSFORMS TO TARGET ###
            self.set_transforms("target")
            transform_opts = self.transform_opts["target"]

        if self.working_dir is not None:
            file = os.path.join(self.working_dir, file_name)
        else:
            file = file_name

        log.debug(f"{self.logfix} File IDX: {idx}, File: {file}")
        sample, _ = self.t_spectrogram(file)
        sample_spec = sample.clone()

        # Data augmentation
        if self.augmentation:
            sample_spec = self.t_amplitude(sample_spec)
            sample_spec = self.t_pitchshift(sample_spec)
            sample_spec = self.t_timestretch(sample_spec)

        sample_orca_detect = sample_spec.clone()
        sample_orca_detect = self.t_compr_a(sample_orca_detect)
        sample_orca_detect = self.t_norm(sample_orca_detect)

        sample_spec, _ = self.sp.detect_strong_spectral_region(
            spectrogram=sample_orca_detect, spectrogram_to_extract=sample_spec, n_fft=self.n_fft,
            target_len=self.seq_len, perc_of_max_signal=self.perc_of_max_signal,
            min_bin_of_interest=int(self.min_thres_detect * sample_orca_detect.shape[-1]),
            max_bin_of_inerest=int(self.max_thres_detect * sample_orca_detect.shape[-1]))

        # Randomly select from a pool of given spectrograms from the strongest regions
        if isinstance(sample_spec, list):
            sample_spec = random.choice(sample_spec).unsqueeze(dim=0)

        sample_spec_ncmpr = sample_spec.clone()

        sample_spec = self.t_compr_f(sample_spec)

        # input not compressed, but 0/1 normalized for binary cases
        binary_input_not_cmpr_not_norm = sample_spec_ncmpr.clone()
        binary_input = self.t_compr_a(binary_input_not_cmpr_not_norm)
        binary_input = self.t_norm(binary_input)

        # frequency compressed, to amplitude and normalized ground truth
        ground_truth = sample_spec.clone()
        ground_truth = self.t_compr_a(ground_truth)
        ground_truth = self.t_norm(ground_truth)

        # ARTF PART

        if self.m is not None:
            distribution_idx = self.m

        if distribution_idx != 0:
            sample_spec = self.t_compr_a(sample_spec)

        if distribution_idx == 4:
            if self.random:
                gaus_stdv = round(random.uniform(0.1, 25.0), 2)
            else:
                gaus_stdv = self.gaus_stdv
            distribution = torch.distributions.normal.Normal(torch.tensor(self.gaus_mean),
                                                             torch.tensor(gaus_stdv)).sample(
                sample_shape=torch.Size([128, 256])).squeeze(dim=-1)
        elif distribution_idx == 1:
            if self.random:
                df = round(random.uniform(0.1, 30.0), 2)
            else:
                df = self.df
            distribution = torch.distributions.chi2.Chi2(torch.tensor(df)).sample(
                sample_shape=torch.Size([128, 256])).squeeze(dim=-1)
        elif distribution_idx == 2:
            if self.random:
                p_lambda = round(random.uniform(0.1, 30.0), 2)
            else:
                p_lambda = self.poisson_lambda
            distribution = torch.distributions.poisson.Poisson(torch.tensor(p_lambda)).sample(
                sample_shape=torch.Size([128, 256])).squeeze(dim=-1)
        elif distribution_idx == 3:
            if self.random:
                e = round(random.uniform(0.05, 0.15), 2)
            else:
                e = self.exp_e
            distribution = torch.distributions.exponential.Exponential(torch.tensor(e)).sample(
                sample_shape=torch.Size([128, 256])).squeeze(dim=-1)
        elif distribution_idx == 0:
            if not self.random:
                self.t_addnoise.min_snr = self.orig_noise_value
                self.t_addnoise.max_snr = self.orig_noise_value
            self.orig_noise = True
        elif distribution_idx == 5:
            # histogram equalization is always constant - no probabilistic effects!
            self.clone = True
        elif distribution_idx == 6:
            self.binary_orig = True
        elif distribution_idx == 7:
            if self.random:
                bin_pow = round(random.uniform(1.3, 2.7), 2)
            else:
                bin_pow = self.bin_pow
            self.binary_ones_pow = True
        elif distribution_idx == 8:
            if self.random:
                bin_pow = round(random.uniform(1.3, 2.7), 2)
            else:
                bin_pow = self.bin_pow
            self.binary_orig_pow = True
        elif distribution_idx == 9:
            self.binary_mask = True

        if self.orig_noise:
            # Add original noise to the sample
            sample_spec_n, _ = self.t_addnoise(sample_spec)
            sample_spec_n = self.t_compr_a(sample_spec_n)
            sample_spec_n = self.t_norm(sample_spec_n)
        elif self.clone:
            sample_spec_n = self.t_compr_a(sample_spec_ncmpr)
            sample_spec_n = self.t_norm(sample_spec_n)
            sample_spec_n = self.sp.search_maxima_spec(sample_spec_n, radius=2)
            sample_spec_n = torch.tensor(exposure.equalize_hist(np.nan_to_num(sample_spec_n.squeeze(dim=0).numpy())),
                                         dtype=torch.float)
            sample_spec_n = self.t_compr_f(sample_spec_n.unsqueeze(dim=0))
        elif self.binary_orig:
            # amplitude and normalized
            sample_spec_n = binary_input.clone()
            binary_mask = self.sp.create_mask(
                sr=transform_opts["sr"],
                fmax=transform_opts["f_max"],
                fmin=transform_opts["f_min"],
                nfft=transform_opts["n_fft"],
                trainable_spectrogram=sample_spec_n
            ).unsqueeze(dim=0)
            ground_truth = binary_input * binary_mask
            ground_truth = self.t_compr_f(ground_truth)
            sample_spec_n = self.t_compr_f(binary_input_not_cmpr_not_norm)
            sample_spec_n = self.t_compr_a(sample_spec_n)
            sample_spec_n = self.t_norm(sample_spec_n)
        elif self.binary_ones_pow:
            sample_spec_n = binary_input.clone()
            binary_mask = self.sp.create_mask(
                sr=transform_opts["sr"],
                fmax=transform_opts["f_max"],
                fmin=transform_opts["f_min"],
                nfft=transform_opts["n_fft"],
                trainable_spectrogram=sample_spec_n).unsqueeze(dim=0)
            ground_truth = binary_input + binary_mask
            ground_truth[ground_truth >= 1.0] = 1.0
            ground_truth = ground_truth.pow(bin_pow)
            ground_truth = self.t_compr_f(ground_truth)
            sample_spec_n = self.t_compr_f(binary_input_not_cmpr_not_norm)
            sample_spec_n = self.t_compr_a(sample_spec_n)
            sample_spec_n = self.t_norm(sample_spec_n)
        elif self.binary_orig_pow:
            sample_spec_n = binary_input.clone()
            binary_mask = self.sp.create_mask(
                sr=transform_opts["sr"],
                fmax=transform_opts["f_max"],
                fmin=transform_opts["f_min"],
                nfft=transform_opts["n_fft"],
                trainable_spectrogram=sample_spec_n).unsqueeze(dim=0)
            ground_truth = binary_input + binary_mask
            ground_truth[ground_truth >= 1.0] = 0.0
            ground_truth = ground_truth.pow(bin_pow)
            ground_truth = ground_truth + (sample_spec_n * binary_mask)
            ground_truth = self.t_compr_f(ground_truth)
            sample_spec_n = self.t_compr_f(binary_input_not_cmpr_not_norm)
            sample_spec_n = self.t_compr_a(sample_spec_n)
            sample_spec_n = self.t_norm(sample_spec_n)
        elif self.binary_mask:
            sample_spec_n = binary_input.clone()
            binary_mask = self.sp.create_mask(
                sr=transform_opts["sr"],
                fmax=transform_opts["f_max"],
                fmin=transform_opts["f_min"],
                nfft=transform_opts["n_fft"],
                trainable_spectrogram=sample_spec_n).unsqueeze(dim=0)
            ground_truth = binary_mask
            ground_truth = self.t_compr_f(ground_truth)
            sample_spec_n = self.t_compr_f(binary_input_not_cmpr_not_norm)
            sample_spec_n = self.t_compr_a(sample_spec_n)
            sample_spec_n = self.t_norm(sample_spec_n)
        else:
            distribution_factor = 1.0
            if not self.do_masking:
                ## Crank up artificial noise by random factor
                distribution_factor = random.randint(1, 3)
            sample_spec_n = sample_spec + (distribution_factor * distribution)
            sample_spec_n = self.t_norm(sample_spec_n)

        label = self.load_label(file)

        label["ground_truth"] = ground_truth
        label["file_name"] = label["file_name"].replace(label["file_name"].rsplit("/", 1)[1],
                                                        self.get_dist_label(distribution_idx)+ "_" + label["file_name"].rsplit("/", 1)[
                                                            1])

        return sample_spec_n, label

    """
    Generate label dict containing filename and whether it is a target signal (call)
    or a noise signal (noise)
    """

    def load_label(self, file_name: str):
        label = dict()
        label["file_name"] = file_name
        label["call"] = True
        return label

    def get_dist_label(self, label):
        if label == 0:
            return "real"
        elif label == 1:
            return "Chi2"
        elif label == 2:
            return "Pois"
        elif label == 3:
            return "Exp"
        elif label == 4:
            return "Gauss"
        elif label == 5:
            return "Hist"
        elif label == 6:
            return "BinXOrig"
        elif label == 7:
            return "Bin1sPow"
        elif label == 8:
            return "BinOPow"
        else:
            return "Bin"


    def get_sample_with_augmentation(self, sample_idx, augmentation):
        self.m = augmentation
        return self.__getitem__(sample_idx)


class HumanSpeechBatchAugmentationDataset(Dataset):
    def __init__(self,
                 split,
                 opts,
                 augmentation,
                 file_names: Iterable[str],
                 noise_directory: str or None,
                 loc=None,
                 *args,
                 **kwargs):

        super().__init__(split, opts, augmentation, file_names, noise_directory, loc, *args, **kwargs)
        self.logfix = f"[Human Speech Data Augmentation -- {split}]:"
        dataset_opts = opts.dataset
        data_opts = opts.data
        if loc is None or loc not in data_opts:
            dir_opts = data_opts
        else:
            dir_opts = data_opts[loc]

        human_speech = dir_opts.human_speech

        log.info(f"{self.logfix} Found {len(file_names)} file pairs for human speech augmentation")
        self.clean_dir = human_speech.clean
        self.noisy_dir = human_speech.noisy

        self.clean_files = file_names
        self.n_fft = 1024
        self.hop = 110
        self.sr = 44100

        spec_transforms = [
            lambda fn: T.load_audio_file(fn, sr=self.sr),
            T.PreEmphasize(dataset_opts.pre_emphasis),
            T.Spectrogram(self.n_fft, self.hop, center=False),
        ]

        self.t_spectrogram = T.Compose(spec_transforms)

        self.t_subseq = T.PaddedSubsequenceSampler(self.seq_len, dim=1, random=False)
        self.t_compr_f = T.Interpolate(self.n_freq_bins, self.sr, 0, 2500)

    def _prep_file(self, file):
        sample, _ = self.t_spectrogram(file)

        # if self.augmentation:
        #     sample = self.t_amplitude(sample)
        #     sample = self.t_pitchshift(sample)
        #     sample = self.t_timestretch(sample)
        sample = self.t_subseq(sample)
        # sample, _ = self.sp.detect_strong_spectral_region(
        #     spectrogram=sample, spectrogram_to_extract=sample, n_fft=self.n_fft,
        #     target_len=self.seq_len, perc_of_max_signal=self.perc_of_max_signal,
        #     min_bin_of_interest=int(self.min_thres_detect * sample.shape[-1]),
        #     max_bin_of_inerest=int(self.max_thres_detect * sample.shape[-1]))
        #
        # if isinstance(sample, list):
        #     sample = random.choice(sample).unsqueeze(dim=0)

        sample = self.t_compr_f(sample)
        sample = self.t_compr_a(sample)
        sample = self.t_norm(sample)

        return sample

    def __getitem__(self, sample):
        clean_file = self.clean_files[sample]
        noisy_file = self.noisy_files[sample]

        clean_sample = self._prep_file(clean_file)
        noisy_sample = self._prep_file(noisy_file)
        label = {
            "ground_truth": clean_sample,
            "file_name": self.clean_files[sample]
        }
        return noisy_sample, label

    def __len__(self):
        return len(self.clean_files)


"""
Dataset for processing an audio tape via a sliding window approach using a given
sequence length and hop size.
"""


class StridedAudioDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            file_name,
            sequence_len: int,
            hop: int,
            sr: int = 44100,
            fft_size: int = 4096,
            fft_hop: int = 441,
            n_freq_bins: int = 256,
            freq_compression: str = "linear",
            f_min: int = 200,
            f_max: int = 18000,
            center=True,
            min_max_normalize=False
    ):

        self.hop = hop
        self.center = center
        self.filename = file_name
        self.sequence_len = sequence_len
        self.audio = T.load_audio_file(file_name, sr=sr, mono=True)
        self.n_frames = self.audio.shape[1]

        spec_t = [
            T.PreEmphasize(DefaultSpecDatasetOps["preemphases"]),
            T.Spectrogram(fft_size, fft_hop, center=self.center),
        ]

        self.spec_transforms = T.Compose(spec_t)

        if freq_compression == "linear":
            self.t_compr_f = (T.Interpolate(n_freq_bins, sr, f_min, f_max))
        elif freq_compression == "mel":
            self.t_compr_f = (T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max))
        elif freq_compression == "mfcc":
            t_mel = T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max)
            self.t_compr_f = (T.Compose(t_mel, T.M2MFCC()))
        else:
            raise "Undefined frequency compression"

        self.t_compr_a = T.Amp2Db(min_level_db=DefaultSpecDatasetOps["min_level_db"])

        if min_max_normalize:
            self.t_norm = T.MinMaxNormalize()
        else:
            self.t_norm = T.Normalize(
                min_level_db=DefaultSpecDatasetOps["min_level_db"],
                ref_level_db=DefaultSpecDatasetOps["ref_level_db"],
            )

    def __len__(self):
        full_frames = max(int(ceil((self.n_frames + 1 - self.sequence_len) / self.hop)), 1)
        if (full_frames * self.sequence_len) < self.n_frames:
            full_frames += 1
        return full_frames

    """
    Extracts signal part according to the current and respective position of the given audio file.
    """

    def __getitem__(self, idx):
        start = idx * self.hop

        end = min(start + self.sequence_len, self.n_frames)

        y = self.audio[:, start:end]

        sample_spec, sample_spec_cmplx = self.spec_transforms(y)

        sample_spec_orig = self.t_compr_a(sample_spec)

        sample_spec = self.t_compr_f(sample_spec_orig)

        sample_spec = self.t_norm(sample_spec)

        return sample_spec_orig, sample_spec, sample_spec_cmplx, self.filename

    def __delete__(self):
        self.loader.join()

    def __exit__(self, *args):
        self.loader.join()


"""
Dataset for processing a folder of various audio files
"""


class SingleAudioFolder(AudioDataset):

    def __init__(
            self,
            file_names: Iterable[str],
            working_dir=None,
            cache_dir=None,
            sr=44100,
            n_fft=1024,
            hop_length=512,
            freq_compression="linear",
            n_freq_bins=256,
            f_min=None,
            f_max=18000,
            center=True,
            min_max_normalize=False,
            *args,
            **kwargs
    ):
        super().__init__(file_names, working_dir, sr, *args, **kwargs)
        if self.dataset_name is not None:
            log.info("Init dataset {}...".format(self.dataset_name))

        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.center = center
        self.hop_length = hop_length
        self.freq_compression = freq_compression

        valid_freq_compressions = ["linear", "mel", "mfcc"]

        if self.freq_compression not in valid_freq_compressions:
            raise ValueError(
                "{} is not a valid freq_compression. Must be one of {}",
                format(self.freq_compression, valid_freq_compressions),
            )

        log.debug(
            "Number of test files: {}".format(len(self.file_names))
        )

        spec_transforms = [
            lambda fn: T.load_audio_file(fn, sr=sr),
            T.PreEmphasize(DefaultSpecDatasetOps["preemphases"]),
            T.Spectrogram(n_fft, hop_length, center=self.center)
        ]

        self.file_reader = AsyncFileReader()

        if cache_dir is None:
            self.t_spectrogram = T.Compose(spec_transforms)
        else:
            self.t_spectrogram = T.CachedSpectrogram(
                cache_dir=cache_dir,
                spec_transform=T.Compose(spec_transforms),
                n_fft=n_fft,
                hop_length=hop_length,
                file_reader=self.file_reader,
            )

        if self.freq_compression == "linear":
            self.t_compr_f = T.Interpolate(
                n_freq_bins, sr, f_min, f_max
            )
        elif self.freq_compression == "mel":
            self.t_compr_f = T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max)
        elif self.freq_compression == "mfcc":
            self.t_compr_f = T.Compose(
                T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max), T.M2MFCC()
            )
        else:
            raise "Undefined frequency compression"

        self.t_compr_a = T.Amp2Db(min_level_db=DefaultSpecDatasetOps["min_level_db"])

        if min_max_normalize:
            self.t_norm = T.MinMaxNormalize()
            log.debug("Init min-max-normalization activated")
        else:
            self.t_norm = T.Normalize(
                min_level_db=DefaultSpecDatasetOps["min_level_db"],
                ref_level_db=DefaultSpecDatasetOps["ref_level_db"],
            )
            log.debug("Init 0/1-dB-normalization activated")
        self.t_subseq = T.PaddedSubsequenceSampler(128, dim=1, random=False)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        if self.working_dir is not None:
            file = os.path.join(self.working_dir, file_name)
        else:
            file = file_name

        sample_spec, sample_spec_cmplx = self.t_spectrogram(file)

        sample_spec_orig = self.t_compr_a(sample_spec)

        sample_spec = self.t_compr_f(sample_spec_orig)

        sample_spec = self.t_norm(sample_spec)

        sample_spec = self.t_subseq(sample_spec)
        sample_spec_cmplx = self.t_subseq(sample_spec_cmplx)

        return sample_spec_orig, sample_spec, sample_spec_cmplx, file_name
