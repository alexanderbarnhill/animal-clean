from typing import List, Tuple
import logging as log
from omegaconf import OmegaConf
import os


def get_configuration(configuration_base: str | List[str] | Tuple[str] | None):
    if isinstance(configuration_base, str) and os.path.isfile(configuration_base):
        log.info(f"Loading configuration file {configuration_base}")
        return OmegaConf.load(configuration_base)

    if isinstance(configuration_base, str) and os.path.isdir(configuration_base):
        files = [os.path.join(configuration_base, f) for f in os.listdir(configuration_base) if (".yaml" in f.lower() or ".yml" in f.lower()) and "template" not in f.lower()]

    elif isinstance(configuration_base, list):
        files = configuration_base
    elif isinstance(configuration_base, tuple):
        files = list(configuration_base)
    else:
        raise IOError(f"Unable to determine type of configuration for type {type(configuration_base)}")

    if len(files) == 0:
        raise FileNotFoundError(f"No configuration files found")

    configs = [OmegaConf.load(file) for file in files]
    log.info(f"Merging {len(configs)} configuration files from {configuration_base}")
    return OmegaConf.unsafe_merge(*configs)


def build_configuration(defaults_path: str | List[str] | Tuple[str], species_configuration: str | None=None):
    configuration = get_configuration(defaults_path)
    if species_configuration is not None:
        species_config = get_configuration(species_configuration)
        configuration = OmegaConf.unsafe_merge(configuration, species_config)
    return configuration
