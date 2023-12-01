from utilities.training import get_data_loaders
from utilities.configuration import build_configuration
import os
from tests.test_base import *


def test_get_data_loaders():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=ORCA)
    loaders = get_data_loaders(configuration)
    assert loaders["train"] is not None
    assert loaders["val"] is not None
    assert loaders["test"] is not None


def test_loaders_not_empty():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=ORCA)
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
    train_loader = loaders["test"]
    assert train_loader is not None
    sample = next(iter(train_loader))
    assert sample is not None




