import argparse
import csv

import torch

from utils.data_readers import wmt19_qe_reader
import utils.metrics as mt_metrics

DATASET_FORMATS = {
    "wmt19": wmt19_qe_reader,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labse_functor = mt_metrics.LabseCosineDistanceFunctor(device)
rouge_functor = mt_metrics.RougeMetricFunctor("rougeL")
adj_bleu_functor = mt_metrics.AdjustedBleuFunctor()

AVAILABLE_METRICS = {
    "labse_src_ref": lambda data: labse_functor(data, "src", "ref"),
    "labse_src_tr": lambda data: labse_functor(data, "src", "tr"),
    "labse_tr_ref": lambda data: labse_functor(data, "tr", "ref"),
    "adj_bleu": adj_bleu_functor,
    "bleu": mt_metrics.compute_bleu,
    "rouge": rouge_functor,
}


def compute_metrics(data, required_metrics):
    metric_values = dict()
    for required_metric in required_metrics:
        value = AVAILABLE_METRICS[required_metric](data)
        metric_values[required_metric] = value

    return metric_values


def write_to_output(writer, metric_values, data):
    row_values = list(data.values()) + list(metric_values.values())
    writer.writerow(row_values)


def add_header(writer, metric_names, data_fields):
    columns = data_fields.copy() + metric_names.copy()
    writer.writerow(columns)


def main(args):
    used_metrics = args.metrics.split(",")
    # reader is required to return line_idx, src, tr, and ref
    # other fields can be returned too but they will just be written to output as is
    reader = DATASET_FORMATS[args.dataset_format](args.dataset_path)
    with open(args.output_path, "w") as output_file:
        writer = csv.writer(output_file, dialect="excel-tab")
        has_header = False
        for data in reader:
            if not has_header:
                add_header(writer, used_metrics, list(data.keys()))
                has_header = True

            metric_values = compute_metrics(data, used_metrics)
            write_to_output(writer, metric_values, data)


def validate_args(args):
    for requested_metric in args.metrics.split(","):
        if requested_metric not in AVAILABLE_METRICS.keys():
            raise ValueError(f"Unknown metric: {requested_metric}")

    if args.dataset_format not in DATASET_FORMATS.keys():
        raise ValueError(f"Unknown dataset format: {args.dataset_format}")


if __name__ == "__main___":
    parser = argparse.ArgumentParser(
        description="Compute metrics for every sentence pair in dataset"
    )
    parser.add_argument("metrics", type=str)
    parser.add_argument("dataset_format", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    validate_args(args)

    main(args)
