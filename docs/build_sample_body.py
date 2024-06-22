import os
from glob import glob
import random
import shutil
import soundfile as sf

EVAL_SRC = "/home/alex/git/animal-clean/docs/"
NAME_MAP = {
    "chimp": "Chimpanzee",
    "chimpanzee": "Chimpanzee",
    "bat": "Pygmy Pipistrelle",
    "orca": "Killer Whale",
    "killer-whale": "Killer Whale",
    "parakeet": "Monk Parakeet",
    "monk-parakeet": "Monk Parakeet",
    "pygmy-pipistrelle": "Pygmy pipistrelle",
    "primate": "ComParE Primate",
    "warbler": "Blue-/Golden-Winged Warbler",
    "NOISE": "Noise",
    "noise": "Noise",
    "target": "Target",
    "call": "Call",
    "BWWA": "Blue Winged Warbler",
    "GWWA": "Golden Winged Warbler",
    "OTHER_BIRD": "Other Bird",
    "CALL": "Call",
    "contact": "Contact",
    "alarm": "Alarm",
    "other": "Other",
    "AMBIENT": "Ambient",
    "OTHERANIMAL": "Other Animal",
    "OTHERBAT": "Other Bat",
    "human-augmentation": "Human Augmentation Included",
    "only-target": "Only Target Data"

}


class AnimalCleanSample:
    def __init__(self, audio_in, audio_out, img_in, img_out, aug_type, src_dir):
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.img_in = img_in
        self.img_out = img_out
        self.aug_type = aug_type
        self.src_dir = src_dir
        self.cls = self.get_class()

    def get_class(self):
        cls = os.path.basename(self.audio_in).split("_")[3].split("-")[0]
        if cls not in NAME_MAP:
            return self.audio_in.split(".wav")[0]
        return NAME_MAP[cls]


    def make_example(self):
        return (f"<div class='example'>"
                f"<div class='original'>"
                f"{self.make_img_line(self.img_in)}{self.make_audio_line(self.audio_in)}"
                f"<p>Original Audio</p>"
                f"</div>"
                f"<div class='denoised'>{self.make_img_line(self.img_out)}{self.make_audio_line(self.audio_out)}"
                f"<p>Denoised Audio</p>"
                f"</div></div>")

    # def make_target_examples_block(self):
    #     s = "<div class='target-examples'>"
    #     for t in tuples:
    #         s += make_example(t[0], t[1], t[2], t[3])
    #     s += "</div>"
    #     return s

    # def make_targets_block(self, name, tuples):
    #     s = f"<h4>{name}</h4>" +make_target_examples_block(tuples)
    #     return s

    def make_audio_line(self, file):
        file = file.replace(self.src_dir, '')
        if file[0] == os.sep:
            file = file[1:]
        return f"<audio controls><source src='{file}' type='audio/wav'></audio>"

    def make_img_line(self, file):
        file = file.replace(self.src_dir, '')
        if file[0] == os.sep:
            file = file[1:]
        return f"<img src='{file}' alt='{os.path.basename(file)}' />"

    def write(self):
        return self.make_example()


def do_sample_compare(audio_in, audio_out):
    y, sr = sf.read(audio_in)
    y2, _ = sf.read(audio_out)

    if len(y2) < len(y):
        y = y[:len(y2)]

        sf.write(audio_in, y, sr)
    print("")

def gather_write_samples(t_name, input_data, output_data):
    audio = glob(input_data + "/**/*.wav", recursive=True)
    imgs = glob(input_data + "/**/*.png")
    audio_in = [f for f in audio if "net_in" in f]
    audio_out = [f for f in audio if "net_out" in f]
    img_in = [f for f in imgs if "net_in" in f]
    img_out = [f for f in imgs if "net_out" in f]

    samples = []

    for a in audio_in:
        pi = "_".join(os.path.basename(a).split("_")[:3])
        po = pi.replace("_in", "_out")
        ii = [i for i in img_in if pi in i]
        io = [i for i in img_out if po in i]
        ao = [i for i in audio_out if po in i]
        if len(ii) == 0 or len(io) == 0 or len(ao) == 0:
            continue
        ii = ii[0]
        io = io[0]
        ao = ao[0]
        do_sample_compare(a, ao)
        sample = AnimalCleanSample(
            img_in=ii,
            img_out=io,
            audio_out=ao,
            audio_in=a,
            aug_type=t_name,
            src_dir=output_data
        )
        samples.append(sample)

    return samples


def make_aug_block(name, aug, src, write_target, sample_write_output, f):
    samples = gather_write_samples(aug, src, sample_write_output)
    f.write(f"<h3>{NAME_MAP[aug]}</h3>\n")
    f.write(f"<div class='aug-block'>")
    clss = list(set([f.cls for f in samples]))
    for cls in clss:
        f.write(f"<div class='cls-block'>")
        f.write(f"<h4>{cls}</h4>")

        cls_samples = [sample for sample in samples if sample.cls == cls]

        for sample in cls_samples:
            f.write(sample.write())
        f.write("</div>")
    f.write(f"</div>")



def make_species_block(name, src, write_target, sample_write_output):

    augs = [f for f in os.listdir(src)]
    with open(write_target, "a") as f:
        f.write(f"<div class='example-block'>")
        f.write(f"<h2>{name}</h2>\n")

        for aug in augs:
            make_aug_block(name, aug, os.path.join(src, aug), write_target, sample_write_output, f)
        f.write(f"</div>")
    f.close()


if __name__ == '__main__':
    targets = [f for f in os.listdir(EVAL_SRC)]
    target_out = "/home/alex/git/animal-clean/docs/body.samples.html"
    s_out = ""
    if os.path.isfile(target_out):
        os.remove(target_out)
    for target in targets:
        species = target
        target_dir = os.path.join(EVAL_SRC, species)
        if not os.path.isdir(target_dir):
            print(f"Skipping {target}")
            continue
        make_species_block(NAME_MAP[species], target_dir, target_out, "/home/alex/git/animal-clean/docs")
