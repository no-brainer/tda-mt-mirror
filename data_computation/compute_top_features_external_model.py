import argparse
import csv
import itertools
import multiprocessing
import os

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

TRANSLATIONS_PATH = args.translations_path
LABELS_PATH = args.input_path
OUTPUT_PATH_BASE = args.output_path_base
WORKERS = args.workers

THRESHS = [0.01, 0.05, 0.15, 0.25]
FEATURES = "wcc,scc,sc,b1,avd".split(",")

os.environ["CUDA_VISIBLE_DEVICES"] = ""
pool = multiprocessing.Pool(len(THRESHS))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

MAX_TOKENS = 128

MODEL_NAME = "DeepPavlov/rubert-base-cased"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_LAYERS = 12
N_HEADS = 12

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)


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


labels_df = pd.read_csv(LABELS_PATH, index_col=0)
translations_df = pd.read_csv(TRANSLATIONS_PATH, sep="\t", index_col=0)
df = labels_df.join(translations_df)
del labels_df, translations_df

for row in df.itertuples():
    attn = grab_attention_weights(model, tokenizer, [row.translation], MAX_TOKENS, DEVICE)

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
