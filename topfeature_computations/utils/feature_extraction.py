import networkx as nx
import numpy as np
import ripserplusplus as rpp_py

from utils.attn_extraction import get_attn_scores
import utils.feature_computation as feature


def prepare_bigraph(incidence_mat, symmetric=False):
    """
    builds bipartite graph in networkx from its incidence matrix
    incidence_mat: 2D tensor of size (L, R), where L and R are the sizes of the parts
    """
    l_size, r_size = incidence_mat.shape
    full_inc_mat = np.zeros((l_size + r_size, l_size + r_size))
    full_inc_mat[:l_size, l_size:] = incidence_mat
    if symmetric:
        full_inc_mat += full_inc_mat.T
    return nx.from_numpy_matrix(full_inc_mat, create_using=nx.DiGraph)


def prepare_graph(incidence_mat):
    return nx.from_numpy_matrix(incidence_mat.numpy(), create_using=nx.MultiDiGraph)


def graph_features_from_attn(attn, thresh, used_features="wcc,scc,sc,avd,e,b0,b1", use_bigraph=False):
    features = []
    binarized_weights = (attn > thresh).int()
    g = prepare_bigraph(binarized_weights) if use_bigraph else prepare_graph(binarized_weights)
    for feat_name in used_features.split(","):
        func = getattr(feature, f"count_{feat_name}")
        stats_value = func(g)
        features.extend(stats_value)

    return features


def graph_features_from_model(model, translator_name, src_sentence, lang_pair, head, layer, thresh, 
                              used_features="wcc,scc,sc,avd,e,b0,b1"):
    """
    used_features are separated by comma
        wcc    - weakly connected components
        scc    - strongly connected components
        sc     - simple cycles
        avd    - average vertex degree
        e      - number of edges
        b0, b1 - betti numbers
    """
    src_lang, trg_lang = lang_pair

    attns = get_attn_scores(src_sentence, model, translator_name, src_lang, trg_lang)
    weights = attns[f"decoder.l{layer}"][0, head]

    return graph_features_from_attn(weights, thresh, used_features)


def remove_inf_barcodes(barcode):
    for dim, persistence_pairs in enumerate(barcode):
        if len(persistence_pairs):
            barcode[dim] = persistence_pairs[np.isfinite(persistence_pairs[:, 1])]
    return barcode


def attn_to_ripser_matrix(attn, thresh=0.0):
    attn = (attn > thresh).int() * attn
    attn = 1. - attn
    attn -= np.diag(np.diag(attn))
    return np.minimum(attn.T, attn)


def ripser_features_from_attn(attn, used_features, maxdim=1):
    data = rpp_py.run(f"--dim {maxdim} --format distance", attn_to_ripser_matrix(attn))

    barcode = []
    for i in range(maxdim + 1):
        if len(data[i]):
            arr = data[i].view(np.float32).reshape(data[i].shape + (-1,))
        else:
            arr = np.empty(shape=(0, 0))
        barcode.append(arr)

    barcode = remove_inf_barcodes(barcode)

    features = []
    for feat_data in used_features:
        feat_parts = feat_data.split("_")
        feat_name, dim = feat_parts[0], int(feat_parts[1])
        args = feat_parts[2:]

        func = getattr(feature, f"count_barcode_{feat_name}")
        features.append(func(barcode, dim, *args))

    return features
