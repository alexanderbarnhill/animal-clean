from typing import Any
import os

import lightning as pl
import torch.nn as nn

CONFIGURATION_BASE = "/home/alex/git/animal-clean/configuration"
SPECIES_BASE = os.path.join(CONFIGURATION_BASE, "species")

DEFAULTS = "defaults.yaml"
MONITORING = "monitoring.yaml"

ORCA = os.path.join(SPECIES_BASE, "orca.yaml")
CHIMP = os.path.join(SPECIES_BASE, "chimp.yaml")


class _DummyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = nn.Linear(512, 2)


class DummyModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = _DummyModel()

    def forward(self, x) -> Any:
        return self.model(x)

