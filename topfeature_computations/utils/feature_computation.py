import networkx as nx


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

def count_b1(g, cap=500):
    """
    cycle basis
    """
    return count_stat(g.to_undirected(), nx.cycle_basis, cap)

def count_avd(g):
    """
    average vertex degree
    """
    degree_values = [data[1] for data in g.degree()]
    return sum(degree_values) / len(degree_values)