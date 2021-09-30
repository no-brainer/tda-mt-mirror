import argparse
import csv
import itertools
import multiprocess
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from features_calculation.grab_weights import grab_attention_weights
from utils.feature_extraction import graph_features_from_attn


parser = argparse.ArgumentParser(
    description="Compute graph topological features"
)
parser.add_argument("translations_path", type=str)
parser.add_argument("labels_path", type=str)
parser.add_argument("output_path_base", type=str)
parser.add_argument("workers", type=int)
args = parser.parse_args()

TRANSLATIONS_PATH = args.translations_path
LABELS_PATH = args.labels_path
OUTPUT_PATH_BASE = args.output_path_base
WORKERS = args.workers

THRESHS = [0.01, 0.05, 0.15, 0.25]
FEATURES = "wcc,scc,sc,b1,avd".split(",")

pool = multiprocess.Pool(len(THRESHS))

MAX_TOKENS = 128

MODEL_NAME = "DeepPavlov/rubert-base-cased"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_LAYERS = 12
N_HEADS = 12


labels_df = pd.read_csv(LABELS_PATH, index_col=0)
translations_df = pd.read_csv(TRANSLATIONS_PATH, sep="\t", index_col=0)
df = labels_df.join(translations_df)
del labels_df, translations_df

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True).to(DEVICE)

attns = []
for sent in df.translation:
    attns.append(grab_attention_weights(model, tokenizer, [sent], MAX_TOKENS, DEVICE))
attns = np.swapaxes(np.concatenate(attns, axis=1), 0, 1)

del model, tokenizer
print("Computed attention scores. Shape", attns.shape)

cols = ["line_idx"]
for layer, head, feat in itertools.product(range(N_LAYERS), range(N_HEADS), FEATURES):
    cols.append(f"l{layer}_h{head}_{feat}")

output_files = []
tsv_writers = []
for thresh in THRESHS:
    common_dir, basename = os.path.split(OUTPUT_PATH_BASE)
    basename_parts = basename.split(".")
    path = os.path.join(
        common_dir,
        "{}_thresh{}.{}".format(".".join(basename_parts[:-1]), thresh, basename_parts[-1])
    )
    output_files.append(open(path, "w"))
    tsv_writers.append(csv.writer(output_files[-1], dialect="excel-tab"))
    tsv_writers[-1].writerow(cols)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

for i, row in enumerate(df.itertuples()):
    print(i)
    args = []
    for thresh, layer, head in itertools.product(THRESHS, range(N_LAYERS), range(N_HEADS)):
        attn = torch.tensor(attns[i, layer, head])
        args.append((attn, thresh, ",".join(FEATURES)))
    
    results = pool.starmap(graph_features_from_attn, args)

    for i in range(len(THRESHS)):
        row_data = [row.Index]
        for j in range(i * N_HEADS * N_LAYERS, (i + 1) * N_HEADS * N_LAYERS):
            row_data.extend(results[j])
        tsv_writers[i].writerow(row_data)

for file in output_files:
    file.close()
