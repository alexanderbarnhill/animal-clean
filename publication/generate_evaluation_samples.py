import os
import shutil
from glob import glob

data = {
    "WA": {
        "IN_IMG": "/media/alex/Datasets/04_animal-clean/evaluation/warbler/output/img/in",
        "OUT_IMG": "/media/alex/Datasets/04_animal-clean/evaluation/warbler/output/img/out",
        "IN_AUDIO": "/media/alex/Datasets/04_animal-clean/evaluation/warbler/input_data",
        "OUT_AUDIO": "/media/alex/Datasets/04_animal-clean/evaluation/warbler/output"
    }
}
o_base = "/home/alex/git/animal-clean/docs"
c = 5

if __name__ == '__main__':
    for t in data.keys():
        a_s = glob(data[t]["IN_AUDIO"] + "**/**/*.wav", recursive=True)

        print("")
