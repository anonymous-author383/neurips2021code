"""Module for holding convenient stuff for emBench.

METHODS
-------
load(path)
    loads the pickled object at path and returns it
sample_rows(graph, num_rows):
    provide coordinates for all entries in num_rows number of random rows
"""
import pickle as pkl
import numpy as np
import sklearn.metrics


def load(path):
    """
    Loads the pickled object at path and returns it

    :param path: string filename of pickled object to load
    """

    f = open(path, 'rb')
    obj = pkl.load(f)
    f.close()
    return obj

def sample_rows(graph, num_rows):
    """
    Provide coordinates for all entries in num_rows number of random rows

    :param graph: CsrGraph to choose rows from.
    :param num_rows: Number of rows to sample.
    """
    rows = np.random.randint(0, high=graph.number_of_nodes(), size=num_rows, dtype=np.int32)
    pairs = np.vstack([np.array([u, v]) for u in graph.nodes() for v in rows])
    return pairs

def sample_from_comm(graph, num_rows):
    """
    Samples vertices which are in communities

    :param graph: CsrGraph to choose rows from.
    :param num_rows: Number of rows to sample.
    """
    comms = np.random.randint(0, high=graph._communities.shape[1], size=num_rows, dtype=np.int32)
    nodes = [np.random.choice(np.nonzero(graph._communities[:, c].flatten())[0]) for c in comms]
    pairs = np.vstack([np.array([u, v]) for u in graph.nodes() for v in nodes])
    return pairs
