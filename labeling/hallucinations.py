import argparse
import csv
import logging
import os
import re

import pandas as pd

from common.sentence_dispatcher import SentencePairDispatcher


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="./h.log", 
    encoding="utf-8", 
    level=logging.INFO,
)

parser = argparse.ArgumentParser(description="Path to translations")
parser.add_argument("score_path", type=str)
parser.add_argument("corpus_path", type=str)
parser.add_argument("labeled_path", type=str)
args = parser.parse_args()

pattern1 = re.compile(r"(\b\w{2,})(\s\1){2,}\b")
pattern2 = re.compile(r"([^\W\d_])\1{3,}")

IS_CLEAN = False

scores_df = pd.read_csv(args.score_path, index_col=0, sep="\t")
dispatcher = SentencePairDispatcher(args.corpus_path)

with open(args.labeled_path, "a+") as f:
    writer = csv.writer(f)
    if os.fstat(f.fileno()).st_size == 0:
        writer.writerow(["line_idx", "label"])

    for row in scores_df.itertuples():
        translation = row.translation
        if pattern1.search(translation) is not None:
            src, _ = dispatcher(row.Index)

            if IS_CLEAN:
                is_hallucination = True
            else:
                print(src)
                print(translation)
                is_hallucination = input() == "y"
            
            if is_hallucination:
                logging.info(f"{src} -> {translation}")
                writer.writerow([row.Index, "hallucination"])
                continue

        if pattern2.search(translation) is not None:
            src, _ = dispatcher(row.Index)

            print(src)
            print(translation)
            if input() == "y":
                writer.writerow([row.Index, "hallucination"])
