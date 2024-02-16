from data.audiodataset import Dataset
from data.audiodataset import HumanSpeechBatchAugmentationDataset
from utilities.training import get_data_loaders, get_human_speech_loader
from utilities.configuration import build_configuration
import os
from tests.test_base import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=PARAKEET)
    configuration.training.batch_size = 2
    loader = get_human_speech_loader(configuration)
    sample_count = 10
    base_out = "/media/alex/s1/doc_work/13_ANIMAL-CLEAN/publication/samples/audio_prep/HUMAN"
    for idx in range(sample_count):
        batch = next(iter(loader))
        features, label = batch
        feature = features.squeeze()
        label = label["ground_truth"].squeeze()

        feature = feature.T.detach().cpu().numpy()
        label = label.T.detach().cpu().numpy()

        fig, ax = plt.subplots(dpi=60)
        ax.imshow(feature, origin="lower", interpolation=None)
        plt.axis("off")
        out_file = os.path.join(base_out, f"{idx}_noisy.pdf")
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(dpi=60)
        ax.imshow(label, origin="lower", interpolation=None)
        plt.axis("off")
        out_file = os.path.join(base_out, f"{idx}_ground_truth.pdf")
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
