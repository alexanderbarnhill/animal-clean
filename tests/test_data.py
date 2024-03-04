from data.audiodataset import HumanSpeechBatchAugmentationDataset
from utilities.training import get_data_loaders, get_human_speech_loader
from utilities.configuration import build_configuration
import os
from tests.test_base import *
import matplotlib.pyplot as plt
import torch
import random
from utilities.viewing import convert_tensor_to_PIL


def _show(sample, transpose=True, file_name=None):
    s = sample.clone()
    s = s.detach().cpu().numpy()
    s = s[0]
    if transpose:
        s = s.T

    fig, ax = plt.subplots(dpi=60)
    plt.imshow(s, origin="lower", interpolation=None)

    plt.show()




def test_get_data_loaders():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=ORCA)
    loaders = get_data_loaders(configuration)
    assert loaders["train"] is not None
    assert loaders["val"] is not None
    assert loaders["test"] is not None


def test_loaders_not_empty():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE)
    loaders = get_data_loaders(configuration)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0


def test_get_data_sample():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=ORCA)
    configuration.training.batch_size = 1
    loaders = get_data_loaders(configuration)
    train_loader = loaders["train"]
    assert train_loader is not None
    sample = next(iter(train_loader))
    assert sample is not None


def test_get_human_speech_dataset():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=ORCA)
    configuration.training.batch_size = 2
    loader = get_human_speech_loader(configuration)

    assert loader is not None
    sample = next(iter(loader))
    assert sample is not None


def test_masking_data_selection():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=PARAKEET)
    configuration.training.batch_size = 4
    loaders = get_data_loaders(configuration)
    train_loader = loaders["train"]
    assert train_loader is not None
    sample = next(iter(train_loader))
    assert sample is not None
    x, y = sample
    ground_truth = y["ground_truth"]
    label = y["file_name"]
    for i in range(ground_truth.shape[0]):
        _show(ground_truth[i])
        img = convert_tensor_to_PIL(ground_truth[i], os.path.basename(label[i]))
        img.show()
    for i in range(x.shape[0]):
        _show(x[i])


def test_masking_data_selection_val_test():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=PARAKEET)
    configuration.training.batch_size = 4
    loaders = get_data_loaders(configuration)
    train_loader = loaders["train"]
    assert train_loader is not None
    sample = next(iter(train_loader))
    assert sample is not None
    x, y = sample
    ground_truth = y["ground_truth"]
    for i in range(ground_truth.shape[0]):
        _show(ground_truth[i])
    for i in range(x.shape[0]):
        _show(x[i])


def test_artificial_noise_scaling():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=PARAKEET)
    configuration.training.batch_size = 4
    configuration.dataset.augmentation.noise_masking.masking_probability = 0.0
    configuration.data.local.noise_sources.train = None
    loaders = get_data_loaders(configuration)
    train_loader = loaders["train"]
    assert train_loader is not None
    sample = next(iter(train_loader))
    assert sample is not None
    x, y = sample
    ground_truth = y["ground_truth"]
    for i in range(ground_truth.shape[0]):
        _show(ground_truth[i])
    for i in range(x.shape[0]):
        _show(x[i])



def test_combine_dataloaders():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=CHIMP)
    configuration.training.batch_size = 8
    h_loader = get_human_speech_loader(configuration)
    loaders = get_data_loaders(configuration)
    train_loader = loaders["train"]
    assert h_loader is not None
    assert train_loader is not None
    train_batch = next(iter(train_loader))
    human_batch = next(iter(h_loader))
    x, y = train_batch
    ground_truth = y["ground_truth"]
    h_x, h_y = human_batch
    h_gt = h_y["ground_truth"]

    x_c = torch.concat([x, h_x], dim=0)
    gt_c = torch.concat([ground_truth, h_gt], dim=0)

    x = x_c.clone()
    ground_truth = gt_c.clone()

    idxs = list(range(0, x.shape[0]))
    random.shuffle(idxs)

    for i, idx in enumerate(idxs):
        x[i] = x_c[idx]
        ground_truth[i] = gt_c[idx]

    for i in idxs:
        img = convert_tensor_to_PIL(image_tensor=ground_truth[i])
        img.show()


    assert x.shape[0] == configuration.training.batch_size
    assert ground_truth.shape[0] == configuration.training.batch_size







