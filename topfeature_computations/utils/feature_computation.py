import networkx as nx
import numpy as np


# GRAPH features

def count_stat(g, func, cap=500):
    result = 0
    for _ in func(g):
        result += 1
        if result >= cap:
            break
    return result


def count_wcc(g, cap=500):
    """
    weakly connected components
    """
    return count_stat(g, nx.weakly_connected_components, cap)


def count_scc(g, cap=500):
    """
    strongly connected components
    """
    return count_stat(g, nx.strongly_connected_components, cap)


def count_sc(g, cap=500):
    """
    simple cycles
    """
    return count_stat(g, nx.simple_cycles, cap)


def count_avd(g, cap=500):
    """
    average vertex degree
    """
    degree_values = [data[1] for data in g.degree()]
    return sum(degree_values) / len(degree_values)


def count_e(g, cap=500):
    return g.number_of_edges()


def count_b0(g, cap=500):
    g_undir = nx.Graph(g.to_undirected())
    w = nx.number_connected_components(g_undir)
    return w


def count_b1(g, cap=500):
    g_undir = nx.Graph(g.to_undirected())
    w = nx.number_connected_components(g_undir)
    e = g_undir.number_of_edges()
    v = g_undir.number_of_nodes()
    return e - v + w


# BARCODE feature

def count_barcode_sum(barcode, dim):
    if not len(barcode[dim]):
        return 0.
    return np.sum(barcode[dim][:, 1] - barcode[dim][:, 0])


def count_barcode_mean(barcode, dim):
    if not len(barcode[dim]):
        return 0.
    return np.mean(barcode[dim][:, 1] - barcode[dim][:, 0])


def count_barcode_std(barcode, dim):
    if not len(barcode[dim]):
        return 0.
    return np.std(barcode[dim][:, 1] - barcode[dim][:, 0])


def count_barcode_entropy(barcode, dim):
    if not len(barcode[dim]):
        return 0.
    lens = barcode[dim][:, 1] - barcode[dim][:, 0]
    return -np.sum(lens * np.log(np.clip(lens, 1e-9, None)))


def count_barcode_number(barcode, dim, thresh):
    thresh = float(thresh)
    if not len(barcode[dim]):
        return 0
    lens = barcode[dim][:, 1] - barcode[dim][:, 0]
    return np.sum(lens > thresh)
