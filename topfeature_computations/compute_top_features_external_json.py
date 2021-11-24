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
from utils.feature_extraction import graph_features_from_attn, ripser_features_from_attn
from utils.data import wikihades


parser = argparse.ArgumentParser(
    description="Compute graph topological features"
)
parser.add_argument("data_path", type=str)
parser.add_argument("output_path_base", type=str)
parser.add_argument("external_model_name", type=str)
parser.add_argument("data_format", type=str)
args = parser.parse_args()

OUTPUT_PATH_BASE = args.output_path_base

THRESHS = [0.01, 0.05, 0.15, 0.25]
FEATURES = ["wcc", "scc", "sc", "b1", "avd"]
RIPSER_FEATURES = [
    "sum_0", "sum_1", 
    "mean_0", "mean_1", 
    "std_0", "std_1", 
    "entropy_0", "entropy_1", 
    "number_0_0", "number_0_0",
]

pool = multiprocess.Pool(len(THRESHS))

MAX_TOKENS = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_LAYERS = 12
N_HEADS = 12


def unpack_features(feats):
    cols = []
    for layer, head, feat in itertools.product(range(N_LAYERS), range(N_HEADS), feats):
        cols.append(f"l{layer}_h{head}_{feat}")
    return cols

def create_writers():
    cols = ["line_idx"] + unpack_features(FEATURES)
    ripser_cols = ["line_idx"] + unpack_features(RIPSER_FEATURES)

    output_files = []
    tsv_writers = []

    common_dir, basename = os.path.split(OUTPUT_PATH_BASE)
    basename_parts = basename.split(".")
    basename = ".".join(basename_parts[:-1])
    basename_ext = basename_parts[-1]

    for thresh in THRESHS:
        path = os.path.join(common_dir, f"{basename}_thresh{thresh}.{basename_ext}")
        output_files.append(open(path, "w"))
        tsv_writers.append(csv.writer(output_files[-1], dialect="excel-tab"))
        tsv_writers[-1].writerow(cols)

    ripser_path = os.path.join(common_dir, f"{basename}_ripser.{basename_ext}")
    output_files.append(open(ripser_path, "w"))
    tsv_writers.append(csv.writer(output_files[-1], dialect="excel-tab"))
    tsv_writers[-1].writerow(ripser_cols)

    return tsv_writers, output_files


tokenizer = AutoTokenizer.from_pretrained(args.external_model_name, do_lower_case=True)
model = AutoModel.from_pretrained(args.external_model_name, output_attentions=True).to(DEVICE)

reader = None
if args.data_format == "wikihades":
    reader = wikihades
else:
    raise ValueError(f"Unknown data format: {args.data_format}")

tsv_writers, output_files = create_writers()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

for data in reader(args.data_path):
    attns = np.squeeze(grab_attention_weights(model, tokenizer, [data["text"]], MAX_TOKENS, DEVICE), axis=1)
    args = []
    for thresh, layer, head in itertools.product(THRESHS, range(N_LAYERS), range(N_HEADS)):
        attn = torch.tensor(attns[layer, head])
        args.append((attn, thresh, ",".join(FEATURES)))
    results = pool.starmap(graph_features_from_attn, args)

    for i in range(len(THRESHS)):
        row_data = [data["line_idx"]]
        for j in range(i * N_HEADS * N_LAYERS, (i + 1) * N_HEADS * N_LAYERS):
            row_data.extend(results[j])
        tsv_writers[i].writerow(row_data)
    
    args = []
    for layer, head in itertools.product(range(N_LAYERS), range(N_HEADS)):
        attn = torch.tensor(attns[layer, head])
        args.append((attn, RIPSER_FEATURES))
    results = pool.starmap(ripser_features_from_attn, args)
    row_data = [data["line_idx"]]
    for data in results:
        row_data.extend(data)
    tsv_writers[-1].writerow(row_data)

for file in output_files:
    file.close()
