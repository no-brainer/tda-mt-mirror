import argparse
from itertools import product
import os
import pickle
import shutil
import sys

import numpy as np
import ripserplusplus as rpp_py
import torch

from feature_src.common.io import select_reader
from feature_src.utils.attn_extraction import get_attn_scores, prepare_for_attn_extraction

sys.path.append("../nmt_model")
from src.models import NMTTransformer
import src.tokenizers
from src.translators import GreedyTranslator
from src.utils import init_obj, parse_config


def prepare_model(config_path, checkpoint_path, **kwargs):
    config = parse_config(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    saved_data = torch.load(checkpoint_path, map_location="cpu")
    model = NMTTransformer(**config["model"])
    model.load_state_dict(saved_data["state_dict"])
    prepare_for_attn_extraction(model)
    model = model.to(device)

    tokenizer = init_obj(src.tokenizers, config["tokenizer"])

    translator = GreedyTranslator(model, tokenizer, device, **kwargs)

    n_layers = config["model"]["num_decoder_layers"]
    n_heads = config["model"]["nhead"]

    return translator, n_layers, n_heads


def compute_and_save_attns(translator, data_format, input_file, tmp_folder, keep_last):
    reader = select_reader(data_format)

    for data in reader(input_file):
        attns = get_attn_scores(data["text"], translator, keep_last)

        if keep_last:
            attn_filename = os.path.join(tmp_folder, f"{data['line_idx']}.npy")
            np.save(attn_filename, attns)
            continue

        with open(os.path.join(tmp_folder, f"{data['line_idx']}.pickle"), "wb") as pickle_file:
            pickle.dump(attns, pickle_file)


def struct_barcode_to_ndarray(struct_barcode):
    maxdim = len(struct_barcode)
    barcode = []
    for i in range(maxdim):
        if len(struct_barcode[i]):
            arr = struct_barcode[i].view(np.float32).reshape(struct_barcode[i].shape + (-1,))
        else:
            arr = np.empty(shape=(0, 0))
        barcode.append(arr)

    return barcode


def compute_diagram_for_attn(attn, max_dim):
    raw_barcode = rpp_py.run(f"--dim {max_dim} --format point-cloud", attn)
    return struct_barcode_to_ndarray(raw_barcode)


def compute_and_save_diagrams(attn_folder, tmp_folder, max_dim):
    barcode_ids = None
    for filename in os.listdir(attn_folder):
        full_filename = os.path.join(attn_folder, filename)
        attns = np.load(full_filename)

        n_layers, n_heads = attns.shape[:2]
        if barcode_ids is None:
            barcode_ids = []
            for layer, head in product(range(n_layers), range(n_heads)):
                barcode_ids.append(f"l{layer}_h{head}")

        results = []
        for layer, head in product(range(n_layers), range(n_heads)):
            barcode = compute_diagram_for_attn(attns[layer, head], max_dim)
            results.append(barcode)

        results = dict(zip(barcode_ids, results))

        result_filename = os.path.join(tmp_folder, filename)
        np.savez(result_filename, **results)


def main(args):
    attn_folder = os.path.join(args.base_path, "attns")
    if not os.path.exists(attn_folder):
        os.makedirs(attn_folder, exist_ok=True)
        translator, _, _ = prepare_model(args.config_path, args.checkpoint_path, max_length=args.max_length)
        compute_and_save_attns(translator, args.data_format, args.data_path, attn_folder, args.keep_last)

        attn_zip_filename = os.path.join(args.base_path, "attns.zip")
        shutil.make_archive(os.path.join(args.base_path, "attns"), "zip", attn_folder)
        print(f"Saved attentions to {attn_zip_filename}")

    if args.skip_diag_computations:
        return

    diag_folder = os.path.join(args.base_path, "diags")
    if not os.path.exists(diag_folder):
        os.makedirs(diag_folder, exist_ok=True)
        compute_and_save_diagrams(attn_folder, diag_folder, args.max_dim)

        diag_zip_filename = os.path.join(args.base_path, "diags.zip")
        shutil.make_archive(os.path.join(args.base_path, "diags"), "zip", diag_folder)
        print(f"Saved persistence diagrams to {diag_zip_filename}")

    shutil.rmtree(attn_folder)
    shutil.rmtree(diag_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute attention maps and persistence diagrams"
    )
    parser.add_argument("data_path", type=str)
    parser.add_argument("data_format", type=str)
    parser.add_argument("base_path", type=str)
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("config_path", type=str)

    parser.add_argument("--max_dim", type=int, default=1)
    parser.add_argument("--max_length", "-l", type=int, default=128)

    parser.add_argument("--collect_all_maps", dest="keep_last", action="store_false")
    parser.add_argument("--skip_diag_computations", action="store_true")

    script_args = parser.parse_args()

    main(script_args)
