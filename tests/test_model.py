from models.model import AnimalClean
from utilities.configuration import get_configuration
from tests.test_base import *
import torch


def test_get_model():
    configuration = get_configuration(CONFIGURATION_BASE)
    model = AnimalClean(configuration)
    assert model is not None


def test_model_path():
    configuration = get_configuration(CONFIGURATION_BASE)
    model = AnimalClean(configuration)
    tensor = torch.randn((1, 1, 128, 256))
    output = model(tensor)
    assert tensor.shape == output.shape
