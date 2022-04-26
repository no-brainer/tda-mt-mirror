import argparse
import csv
import itertools
import multiprocess
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from common.io import create_writers, select_reader
from features_calculation.grab_weights import grab_attention_weights
from utils.feature_extraction import graph_features_from_attn, ripser_features_from_attn


THRESHS = [0.01, 0.05, 0.15, 0.25]
FEATURES = ["wcc", "scc", "sc", "b0", "b1", "avd", "e"]
RIPSER_FEATURES = [
    "sum_0", "sum_1", 
    "mean_0", "mean_1", 
    "std_0", "std_1", 
    "entropy_0", "entropy_1", 
    "number_0_0", "number_0_0",
]

MAX_TOKENS = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_LAYERS = 12
N_HEADS = 12


def compute_graph_features(line_idx, attns, pool, tsv_writers):
    func_args = []
    for thresh, layer, head in itertools.product(THRESHS, range(N_LAYERS), range(N_HEADS)):
        attn = torch.tensor(attns[layer, head])
        func_args.append((attn, thresh, ",".join(FEATURES)))

    results = pool.starmap(graph_features_from_attn, func_args)

    for i in range(len(THRESHS)):
        row_data = [line_idx]
        for j in range(i * N_HEADS * N_LAYERS, (i + 1) * N_HEADS * N_LAYERS):
            row_data.extend(results[j])

        tsv_writers[i].writerow(row_data)


def compute_ripser_features(line_idx, attns, pool, tsv_writers):
    func_args = []
    for layer, head in itertools.product(range(N_LAYERS), range(N_HEADS)):
        attn = attns[layer, head]
        func_args.append((attn, RIPSER_FEATURES))

    results = pool.starmap(ripser_features_from_attn, func_args)

    row_data = [line_idx]
    for data in results:
        row_data.extend(data)

    tsv_writers[-1].writerow(row_data)


def main(args):
    reader = select_reader(args.data_format)

    pool = multiprocess.get_context("spawn").Pool(args.num_workers)

    tsv_writers, output_files = create_writers(
        args.output_path_base, N_LAYERS, N_HEADS, THRESHS, FEATURES, RIPSER_FEATURES
    )

    tokenizer = AutoTokenizer.from_pretrained(args.external_model_name, do_lower_case=True)
    model = AutoModel.from_pretrained(args.external_model_name, output_attentions=True).to(DEVICE)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for data in reader(args.data_path):
        attns = np.squeeze(grab_attention_weights(model, tokenizer, [data["text"]], MAX_TOKENS, DEVICE), axis=1)
        compute_graph_features(data["line_idx"], attns, pool, tsv_writers)
        compute_ripser_features(data["line_idx"], attns, pool, tsv_writers)

    for file in output_files:
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute topological features"
    )
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_path_base", type=str)
    parser.add_argument("external_model_name", type=str)
    parser.add_argument("data_format", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    args = parser.parse_args()

    main(args)
