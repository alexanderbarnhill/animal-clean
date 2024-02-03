from tests.test_base import *
import os
from utilities.configuration import get_configuration, build_configuration


def test_configuration_merge():
    configuration = get_configuration(configuration_base=CONFIGURATION_BASE)
    assert configuration is not None, "Configuration cannot be None"


def test_configuration_merge_list():
    configuration = get_configuration(configuration_base=[
        os.path.join(CONFIGURATION_BASE, DEFAULTS),
        os.path.join(CONFIGURATION_BASE, MONITORING)
    ])

    assert configuration is not None, "Configuration cannot be None"


def test_configuration_merge_tuple():
    configuration = get_configuration(configuration_base=(
        os.path.join(CONFIGURATION_BASE, DEFAULTS),
        os.path.join(CONFIGURATION_BASE, MONITORING)
    ))
    assert configuration is not None, "Configuration cannot be None"


def test_configuration_merge_duplicates():
    configuration = get_configuration(configuration_base=(
        os.path.join(CONFIGURATION_BASE, DEFAULTS),
        os.path.join(CONFIGURATION_BASE, DEFAULTS)
    ))
    assert configuration is not None, "Configuration cannot be None"


def test_configuration_default():
    configuration = get_configuration(
        configuration_base=os.path.join(CONFIGURATION_BASE, DEFAULTS))
    assert configuration is not None, "Configuration cannot be None"


def test_configuration_monitoring():
    configuration = get_configuration(
        configuration_base=os.path.join(CONFIGURATION_BASE, MONITORING))
    assert configuration is not None, "Configuration cannot be None"


def test_configuration_species():
    configuration = build_configuration(
        defaults_path=CONFIGURATION_BASE,
        species_configuration=os.path.join(CONFIGURATION_BASE, f"species{os.sep}ape.yaml"))
    assert configuration.dataset.fft.n_fft == 2048
