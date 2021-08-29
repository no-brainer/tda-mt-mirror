import argparse
import csv
import itertools
import multiprocessing
import os

from easynmt import EasyNMT
import torch

from utils.data import tsv_sentence_pairs
from utils.feature_extraction import graph_features_from_attn
from utils.attn_extraction import get_attn_scores


parser = argparse.ArgumentParser(
    description="Compute graph topological features"
)
parser.add_argument("input_path", type=str)
parser.add_argument("output_path_base", type=str)
parser.add_argument("workers", type=int)
parser.add_argument("--skip_special_tokens", action="store_false")
args = parser.parse_args()

OUTPUT_PATH_BASE = args.output_path_base
DATASET_PATH = args.input_path
WORKERS = args.workers
SKIP_SPECIAL_TOKENS = args.skip_special_tokens

THRESHS = [0.01, 0.05, 0.15, 0.25]
FEATURES = "wcc,scc,sc,b1,avd".split(",")

os.environ["CUDA_VISIBLE_DEVICES"] = ""
pool = multiprocessing.Pool(len(THRESHS))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

TGT_LANG = "rus"
SRC_LANG = "eng"

MODEL_TYPE = "opus-mt"
MODEL_NAME = f"Helsinki-NLP/opus-mt-{SRC_LANG[:2]}-{TGT_LANG[:2]}"

BEAM_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_LAYERS = 6
N_HEADS = 8

model = EasyNMT(MODEL_TYPE)
translator = model.translator.load_model(MODEL_NAME)

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

for line_idx, tgt_sentence, src_sentence in tsv_sentence_pairs(DATASET_PATH, TGT_LANG, SRC_LANG):
    attns = get_attn_scores(src_sentence[0], model, MODEL_NAME, SRC_LANG[:2], TGT_LANG[:2], cut_special_tokens=SKIP_SPECIAL_TOKENS)

    args = []
    for thresh, layer, head in itertools.product(THRESHS, range(N_LAYERS), range(N_HEADS)):
        attn = attns[f"decoder.l{layer}"][0, head]
        args.append((attn, thresh, ",".join(FEATURES)))
    
    results = pool.starmap(graph_features_from_attn, args)

    for i in range(len(THRESHS)):
        row_data = [line_idx[0]]
        for j in range(i * N_HEADS * N_LAYERS, (i + 1) * N_HEADS * N_LAYERS):
            row_data.extend(results[j])
        tsv_writers[i].writerow(row_data)

for file in output_files:
    file.close()