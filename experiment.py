import os

import pandas as pd
from tqdm import tqdm

from renal import Renal


tqdm.pandas()

DATA_PATH = "data/mat/"
METADATA_PATH = "data/dane.tsv"


def process_sample(row):
    segmentation_path = os.path.join(DATA_PATH, row["Path"].replace("\\", "/")[1:])
    renal = Renal(segmentation_path, row["Nerka"], row["Guz"])
    scores = renal.get_all_scores()

    results = {"Pred_" + key: value for key, value in scores.items()}
    return pd.Series(
        results,
    )


if __name__ == "__main__":
    metadata = pd.read_csv(METADATA_PATH, delimiter="\t")
    results_metadata = metadata.progress_apply(process_sample, axis=1)
    total = pd.concat([metadata, results_metadata], axis=1)
    total.to_csv(
        path_or_buf="data/wyniki.tsv",
        sep="\t",
        mode="w+",
    )