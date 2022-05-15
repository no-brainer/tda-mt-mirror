import argparse
from itertools import product
import os
import shutil

import numpy as np
import ripserplusplus as rpp_py
import torch
from transformers import AutoTokenizer, AutoModel

from feature_src.features_calculation.grab_weights import grab_attention_weights
from feature_src.common.io import select_reader


MAX_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_and_save_attns(model_name, data_format, input_file, tmp_folder):
    reader = select_reader(data_format)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = AutoModel.from_pretrained(model_name, output_attentions=True).to(DEVICE)

    for data in reader(input_file):
        attns = np.squeeze(grab_attention_weights(model, tokenizer, [data["text"]], MAX_TOKENS, DEVICE), axis=1)

        attn_filename = os.path.join(tmp_folder, f"{data['line_idx']}.npy")
        np.save(attn_filename, attns)


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
        compute_and_save_attns(args.external_model_name, args.data_format,
                               args.data_path, attn_folder)

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
    parser.add_argument("base_path", type=str)
    parser.add_argument("external_model_name", type=str)
    parser.add_argument("data_format", type=str)
    parser.add_argument("--max_dim", type=int, default=1)
    parser.add_argument("--skip_diag_computations", action="store_true")

    args = parser.parse_args()

    main(args)
