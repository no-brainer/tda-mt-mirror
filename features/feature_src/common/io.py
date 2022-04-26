import csv
import itertools
import os

from feature_src.utils.data_readers import wikihades, wmt19_format, scarecrow_format, custom_dataset_format


def unpack_features(n_layers, n_heads, features):
    cols = []
    for layer, head, feature in itertools.product(range(n_layers), range(n_heads), features):
        cols.append(f"l{layer}_h{head}_{feature}")
    return cols


def create_writers(output_base_path, n_layers, n_heads, threshs, graph_features, ripser_features):
    cols = ["line_idx"] + unpack_features(n_layers, n_heads, graph_features)
    ripser_cols = ["line_idx"] + unpack_features(n_layers, n_heads, ripser_features)

    output_files = []
    tsv_writers = []

    common_dir, basename = os.path.split(output_base_path)
    basename_parts = basename.split(".")
    basename = ".".join(basename_parts[:-1])
    basename_ext = basename_parts[-1]

    for thresh in threshs:
        path = os.path.join(common_dir, f"{basename}_thresh{thresh}.{basename_ext}")
        output_files.append(open(path, "w"))
        tsv_writers.append(csv.writer(output_files[-1], dialect="excel-tab"))
        tsv_writers[-1].writerow(cols)

    ripser_path = os.path.join(common_dir, f"{basename}_ripser.{basename_ext}")
    output_files.append(open(ripser_path, "w"))
    tsv_writers.append(csv.writer(output_files[-1], dialect="excel-tab"))
    tsv_writers[-1].writerow(ripser_cols)

    return tsv_writers, output_files


def select_reader(data_format):
    if data_format == "wikihades":
        reader = wikihades
    elif data_format == "wmt19":
        reader = wmt19_format
    elif data_format == "scarecrow":
        reader = scarecrow_format
    elif data_format == "custom":
        reader = custom_dataset_format
    else:
        raise ValueError(f"Unknown data format: {data_format}")

    return reader
