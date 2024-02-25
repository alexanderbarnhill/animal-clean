import os
import shutil
from glob import glob
import pandas as pd


if __name__ == '__main__':
    in_dir = "/media/alex/Datasets/04_animal-clean/training/data/classification"
    out_dir = "/media/alex/Datasets/04_animal-clean/evaluation"

    targets = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    targets = [t for t in targets if "orca" not in t]
    for target in targets:
        t_name = target.split("/")[-1]
        print("Processing {}".format(t_name))
        test_csvs = glob(target + "/**/**/test.csv", recursive=True)
        print(test_csvs)
        for test_csv in test_csvs:
            df = pd.read_csv(test_csv, names=["file"])
            print("Processing {}".format(test_csv))
            class_name = test_csv.split("/")[-2]
            target_out = os.path.join(out_dir, os.path.join(t_name, class_name))
            os.makedirs(target_out, exist_ok=True)
            df["path"] = df["file"].map(lambda x: os.path.join(target, x))
            sampled = df.sample(n=min(len(df), 5))
            for row in sampled.itertuples():
                shutil.copy(row.path, target_out)

