import argparse
import multiprocess
import sys

import torch

from feature_src.common.computations import compute_ripser_features, compute_graph_features
from feature_src.common.io import create_writers, select_reader
from feature_src.utils.attn_extraction import get_attn_scores, prepare_for_attn_extraction

sys.path.append("../nmt_model")
from src.models import NMTTransformer
import src.tokenizers
from src.translators import GreedyTranslator
from src.utils import init_obj, parse_config


THRESHS = [0.01, 0.05, 0.15, 0.25]
FEATURES = ["wcc", "scc", "sc", "b0", "b1", "avd", "e"]
RIPSER_FEATURES = [
    "sum_0", "sum_1",
    "mean_0", "mean_1",
    "std_0", "std_1",
    "entropy_0", "entropy_1",
    "number_0_0", "number_0_0",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_model(config_path, checkpoint_path, **kwargs):
    config = parse_config(config_path)

    saved_data = torch.load(checkpoint_path, map_location="cpu")
    model = NMTTransformer(**config["model"])
    model.load_state_dict(saved_data["state_dict"])
    prepare_for_attn_extraction(model)
    model = model.to(DEVICE)

    tokenizer = init_obj(src.tokenizers, config["tokenizer"])

    translator = GreedyTranslator(model, tokenizer, DEVICE, **kwargs)

    n_layers = config["model"]["num_decoder_layers"]
    n_heads = config["model"]["nhead"]

    return translator, n_layers, n_heads


def main(args):
    reader = select_reader(args.data_format)

    pool = multiprocess.get_context("spawn").Pool(args.num_workers)

    translator, n_layers, n_heads = prepare_model(
        args.config_path,
        args.checkpoint_path,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        pad_id=args.pad_id,
        max_length=args.max_length
    )

    tsv_writers, output_files = create_writers(
        args.output_path_base, n_layers, n_heads, THRESHS, FEATURES, RIPSER_FEATURES
    )

    for data in reader(args.data_path):
        attns = get_attn_scores(data["text"], translator)
        compute_graph_features(data["line_idx"], THRESHS, attns, pool, tsv_writers, FEATURES)
        compute_ripser_features(data["line_idx"], attns, pool, tsv_writers, RIPSER_FEATURES)

    for file in output_files:
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute topological features for MT model")
    parser.add_argument("data_path", type=str)
    parser.add_argument("data_format", type=str)
    parser.add_argument("output_path_base", type=str)
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("config_path", type=str)
    parser.add_argument("--num_workers", default=1, type=int)

    parser.add_argument("--max_length", "-l", type=int, default=128)
    parser.add_argument("--bos_id", type=int, default=2)
    parser.add_argument("--eos_id", type=int, default=3)
    parser.add_argument("--pad_id", type=int, default=0)

    script_args = parser.parse_args()

    main(script_args)
