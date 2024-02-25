import os
from glob import glob
import random
import shutil

EVAL_SRC = "/media/alex/Datasets/04_animal-clean/evaluation"
NAME_MAP = {
    "chimp": "Chimpanzee",
    "bat": "Pygmy Pipistrelle",
    "orca": "Killer Whale",
    "parakeet": "Monk Parakeet",
    "primate": "ComParE Primate",
    "warbler": "Blue-/Golden-Winged Warbler",
    "NOISE": "Noise",
    "BWWA": "Blue Winged Warbler",
    "GWWA": "Golden Winged Warbler",
    "OTHER_BIRD": "Other Bird",
    "CALL": "Call",
    "contact": "Contact",
    "alarm": "Alarm",
    "other": "Other",
    "AMBIENT": "Ambient",
    "OTHERANIMAL": "Other Animal",
    "OTHERBAT": "Other Bat"
}
def make_example(img_in, img_out, audio_in, audio_out):
    return f"<div class='example'><div class='original'>{make_img_line(img_in)}{make_audio_line(audio_in)}<p>Original Audio</p></div><div class='denoised'>{make_img_line(img_out)}{make_audio_line(audio_out)}<p>Denoised Audio</p></div></div>"

def make_target_examples_block(tuples):
    s = "<div class='target-examples'>"
    for t in tuples:
        s += make_example(t[0], t[1], t[2], t[3])
    s += "</div>"
    return s

def make_targets_block(name, tuples):
    s = f"<h4>{name}</h4>" + make_target_examples_block(tuples)
    return s

def make_audio_line(file):
    return f"<audio controls><source src='{file}' type='audio/wav'></audio>"

def make_img_line(file):
    return f"<img src='{file}' />"

def make_species_block(name, src, write_target, sample_write_output):
    input_data = os.path.join(src, "input_data")
    output_data = os.path.join(src, "output")
    if not os.path.isdir(input_data) or not os.path.isdir(output_data):
        print(f"Skipping {name}")
        return

    classes = [f for f in os.listdir(input_data)]
    classes = sorted(classes)
    if "NOISE" in classes:
        c_id = classes.index("NOISE")
        classes.pop(c_id)
        classes.append("NOISE")
    with open(write_target, "a") as f:
        f.write("<div class='example-block'>")
        f.write(f"<h3>{NAME_MAP[name]}</h3>")
        for t in classes:
            f.write(f"<div class='targets'><h4>{NAME_MAP[t]}</h4>{make_target_examples_block(gather_write_sample_tuples(name, t, input_data, output_data, sample_write_output))}</div>")
        f.write("</div>")
        f.close()
    pass

def gather_write_sample_tuples(t_name, c_name, input_data, output_data, output_dir):
    in_audio = glob(input_data + "**/**/*.wav", recursive=True)
    in_audio = [f for f in in_audio if c_name in f]
    imgs = glob(output_data + "**/**/*.png", recursive=True)
    out_imgs = [i for i in imgs if "net_out" in i]
    in_imgs = [i for i in imgs if "net_in" in i]
    out_audio = glob(output_data + "**/**/*.wav", recursive=True)
    bases = [os.path.basename(f).replace(".wav", "") for f in in_audio]
    tuples = []

    source_base_out = os.path.join(t_name, c_name)
    audio_base_out = os.path.join(source_base_out, "AUDIO")
    img_base_out = os.path.join(source_base_out, "IMG")

    audio_original_out = os.path.join(audio_base_out, "ORIGINAL")
    audio_denoised_out = os.path.join(audio_base_out, "DENOISED")
    img_original_out = os.path.join(img_base_out, "ORIGINAL")
    img_denoised_out = os.path.join(img_base_out, "DENOISED")

    audio_original_out_full = os.path.join(output_dir, audio_original_out)
    audio_denoised_out_full = os.path.join(output_dir, audio_denoised_out)
    img_original_out_full = os.path.join(output_dir, img_original_out)
    img_denoised_out_full = os.path.join(output_dir, img_denoised_out)

    if os.path.isdir(audio_original_out_full):
        shutil.rmtree(audio_original_out_full)

    if os.path.isdir(audio_denoised_out_full):
        shutil.rmtree(audio_denoised_out_full)

    if os.path.isdir(img_original_out_full):
        shutil.rmtree(img_original_out_full)

    if os.path.isdir(img_denoised_out_full):
        shutil.rmtree(img_denoised_out_full)

    os.makedirs(audio_original_out_full)
    os.makedirs(audio_denoised_out_full)
    os.makedirs(img_original_out_full)
    os.makedirs(img_denoised_out_full)

    for i in random.sample(in_audio, min(5, len(in_audio))):
        b = os.path.basename(i).replace(".wav", "")
        idx = bases.index(b)

        audio_in = in_audio[idx]
        img_in = None
        img_out = None
        audio_out = None

        for img in in_imgs:
            ib = os.path.basename(img).replace(".png", "")
            if b in ib:
                img_in = img
        for img in out_imgs:
            ib = os.path.basename(img).replace(".png", "")
            if b in ib:
                img_out = img

        for img in out_audio:
            ib = os.path.basename(img).replace(".wav", "")
            if b in ib:
                audio_out = img
        if img_in is None or img_out is None or audio_out is None:
            print(f"Skipping {audio_in}")
            continue


        shutil.copy(audio_in, audio_original_out_full)
        shutil.copy(audio_out, audio_denoised_out_full)
        shutil.copy(img_in, img_original_out_full)
        shutil.copy(img_out, img_denoised_out_full)

        tuples.append((
            os.path.join(img_original_out, os.path.basename(img_in)),
            os.path.join(img_denoised_out, os.path.basename(img_out)),
            os.path.join(audio_original_out, os.path.basename(audio_in)),
            os.path.join(audio_denoised_out, os.path.basename(audio_out)),

        ))



    return tuples


if __name__ == '__main__':
    targets = [f for f in os.listdir(EVAL_SRC)]
    target_out = "/home/alex/git/animal-clean/docs/body.samples.html"
    s_out = ""
    if os.path.isfile(target_out):
        os.remove(target_out)
    for target in targets:
        species = target
        target_dir = os.path.join(EVAL_SRC, species)
        make_species_block(species, target_dir, target_out, "/home/alex/git/animal-clean/docs")
