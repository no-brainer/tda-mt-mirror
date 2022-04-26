from itertools import product

import torch

from feature_src.utils.feature_extraction import graph_features_from_attn, ripser_features_from_attn


def compute_graph_features(line_idx, threshs, attns, pool, tsv_writers, features):
    n_layers, n_heads = attns.shape[:2]
    func_args = []
    for thresh, layer, head in product(threshs, range(n_layers), range(n_heads)):
        attn = torch.tensor(attns[layer, head])
        func_args.append((attn, thresh, ",".join(features)))

    results = pool.starmap(graph_features_from_attn, func_args)

    for i in range(len(threshs)):
        row_data = [line_idx]
        for j in range(i * n_heads * n_layers, (i + 1) * n_heads * n_layers):
            row_data.extend(results[j])

        tsv_writers[i].writerow(row_data)


def compute_ripser_features(line_idx, attns, pool, tsv_writers, features):
    n_layers, n_heads = attns.shape[:2]

    func_args = []
    for layer, head in product(range(n_layers), range(n_heads)):
        attn = attns[layer, head]
        func_args.append((attn, features))

    results = pool.starmap(ripser_features_from_attn, func_args)

    row_data = [line_idx]
    for data in results:
        row_data.extend(data)

    tsv_writers[-1].writerow(row_data)
