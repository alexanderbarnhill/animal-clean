from tests.test_base import *
from utilities.configuration import build_configuration
from utilities.training import get_run_name
def test_get_name():
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=ORCA)
    name = get_run_name(configuration, "cluster")
    assert name is not None