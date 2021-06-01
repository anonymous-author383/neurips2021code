"""Module to do walks for deepwalk

METHODS
-------
do_walks(graph, walk_length, walk_number)
    performs walks and returns iterator over walk transcripts
"""
from numba import njit, prange
from numpy import random
import numpy as np


@njit()
def do_step(v, rows, indxs):
    """does one step of a walk from v

    PARAMETERS
    ----------
    v : int
        vertex from which to step
    rows : np.ndarray
        array containing all rows of adjacency matrix concatenated
    indxs : np.ndarray
        array of pointers into rows

    RETURNS
    _______
    int
        next step in random walk
    """
    return random.choice(rows[indxs[v]:indxs[v + 1]])


@njit(parallel=True)
def do_walk(rows, indxs, num_steps, endpoints, walks):
    """
    does a walk from every vertex given in endpoints

    PARAMETERS
    ----------
    rows : np.ndarray
        array containing column indices of all nonzero coordinates
        in the adjacency matrix
    indxs : np.ndarray
        array of pointers into rows indicating the start of each row
    num_steps : int
        length of walk to perform
    endpoints : np.ndarray
        array of endpoints from which to start walks
    walks : np.ndarray
        empty placeholder array which will be filled with walk transcripts

    RETURNS
    _______
    np.ndarray containing walk transcripts
    """
    walks[:, 0] = endpoints
    for v in prange(len(endpoints)):
        for j in range(1, num_steps):
            walks[v, j] = do_step(walks[v, j - 1], rows, indxs)
    return walks


def process_graph(g):
    rows = g.row_indxs
    indxs = g.row_ptrs
    return rows, indxs


class _WalkContainer():
    """Iterator containing the walk transcripts"""

    def __init__(self, height, width):
        """height, width: ints for size of walk container"""
        self.walks = np.zeros((height, width), dtype=np.int32)

    def __iter__(self):
        for walk in self.walks:
            yield [str(x) for x in walk]


def do_walks(graph, walk_length, walk_number):
    """
    Perform random walks


    PARAMETERS
    ----------
    graph : CsrGraph
        graph to do walks on
    walk_length : int
        length of random walks
    walk_number : int
        number of random walks per vertex

    RETURNS
    -------
    iterator containing walk as lists of strings
    """
    rows, indxs = process_graph(graph)
    n = len(indxs) - 1
    walk_container = _WalkContainer(n * walk_number, walk_length)
    for i in range(walk_number):
        endpoints = np.arange(n, dtype=np.int32)
        do_walk(rows, indxs, walk_length, endpoints,
                walk_container.walks[i * n:(i + 1) * n])
    return walk_container
