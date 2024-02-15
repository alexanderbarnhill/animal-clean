from data.audiodataset import Dataset
from data.audiodataset import HumanSpeechBatchAugmentationDataset
from utilities.training import get_data_loaders, get_human_speech_loader
from utilities.configuration import build_configuration
import os
from tests.test_base import *
import matplotlib.pyplot as plt

mappings = {
    0: "Additive",
    1: "Chi2",
    2: "Poisson",
    3: "Exponential",
    4: "Gaussian",
    5: "Histogram",
    6: "Binary Original",
    7: "Binary Ones",
    8: "Binary Original Potentiated",
    9: "Binary Mask"
}

if __name__ == '__main__':
    sample_count = 10
    output_base = "/media/alex/s1/doc_work/13_ANIMAL-CLEAN/publication/samples/audio_prep"
    d = [

        {
            "c": PARAKEET,
            "m": "MP"
        },

    ]


    for e in d:
        configuration = build_configuration(defaults_path=CONFIGURATION_BASE, species_configuration=e["c"])
        configuration.training.batch_size = 1
        loaders = get_data_loaders(configuration)
        train_loader = loaders["train"]
        dataset = train_loader.dataset
        for m in range(10):

            s_out = os.path.join(output_base, e["m"], mappings[m])
            gt_out = os.path.join(output_base, e["m"], "groundtruth")
            os.makedirs(s_out, exist_ok=True)
            os.makedirs(gt_out, exist_ok=True)


            for idx in range(sample_count):
                sample = dataset.get_sample_with_augmentation(idx, m)
                sample, label = sample
                sample = sample.squeeze()
                sample = sample.T

                gt = label["ground_truth"]
                gt = gt.squeeze()
                gt = gt.T
                gt = gt.detach().cpu().numpy()
                fig, ax = plt.subplots(dpi=60)
                ax.imshow(gt, origin="lower", interpolation=None)
                plt.axis("off")
                out_file = os.path.join(gt_out, f"{idx}-{mappings[m]}_{os.path.basename(label['file_name'])}.pdf")
                plt.savefig(out_file, bbox_inches="tight")
                plt.close(fig)

                sample = sample.detach().cpu().numpy()
                fig, ax = plt.subplots(dpi=60)
                ax.imshow(sample, origin="lower", interpolation=None)
                plt.axis("off")
                out_file = os.path.join(s_out, f"{idx}-{os.path.basename(label['file_name'])}.pdf")
                plt.savefig(out_file, bbox_inches="tight")
                plt.close(fig)