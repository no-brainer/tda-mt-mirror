import multiprocessing
from ripser import ripser

import networkx as nx
from networkx.algorithms import bipartite
import numpy as np

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
    return nx.from_numpy_matrix(incidence_mat.numpy(), create_using=nx.DiGraph)

def graph_features_from_attn(attn, thresh, used_features="wcc,scc,sc,b1,avd", use_bigraph=False):
    features = []
    binarized_weights = (attn > thresh).int()
    g = prepare_bigraph(binarized_weights) if use_bigraph else prepare_graph(binarized_weights)
    for feat_name in used_features.split(","):
        func = getattr(feature, f"count_{feat_name}")
        features.append(func(g))
    return features

def graph_features_from_model(model, translator_name, src_sentence, lang_pair, head, layer, thresh, 
                        used_features="wcc,scc,sc,b1,avd"):
    """
    used_features are separated by comma
        wcc - weakly connected components
        scc - strongly connected components
        sc  - simple cycles
        b1  - cycle basis
        avd - average vertex degree
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

def ripser_features_from_attn(attn, used_features, maxdim=1, metric="euclidean"):
    data = ripser(attn, maxdim=maxdim, metric=metric)
    barcode = data["dgms"]
    barcode = remove_inf_barcodes(barcode)

    features = []
    for feat_data in used_features:
        feat_parts = feat_data.split("_")
        feat_name, dim = feat_parts[0], int(feat_parts[1])
        args = feat_parts[2:]

        func = getattr(feature, f"count_barcode_{feat_name}")
        features.append(func(barcode, dim, *args))

    return features
