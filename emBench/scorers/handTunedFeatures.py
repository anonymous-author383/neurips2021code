"""
Module for holding the hand-tuned scorer's feature calculations

METHODS
-------
cosine_sim(pairs, graph)
    computes cosine similarities between pairs
    and right_adj_vectors
ppr_preprocess(graph, alpha=0.8, tol=0.01, workers=20)
    computes the approximate PPR matrix
ppr_cluster(G, Gvol, seed, alpha=0.99, tol=0.0001)
    computes an approximate ppr vector for a given seed set
compute_three_paths(pairs, graph)
    computes number of paths of length 3
"""
from numba import njit, prange
from scipy import sparse
from multiprocessing import Pool
import numpy as np
import collections


def cosine_sim(pairs, m):
    """
    Computes cosine similarities between all pairs of
    vertices in pairs.

    PARAMETERS
    ----------
    pairs : np.ndarray(dtype=np.int32, shape=(k, 2))
        array of pairs to compute cosine similarities for
    m : scipy.sparse.csr_matrix
        csr_matrix representation of graph to use to
        compute cosine similarities

    RETURNS
    -------
    np.ndarray(dtype=np.float32, shape=(k,1))
        array of cosine similarities
    """
    left_adj_vectors = m[pairs[:, 0]]
    right_adj_vectors = m[pairs[:, 1]]
    lav_ptr = left_adj_vectors.indptr
    lav_col = left_adj_vectors.indices
    rav_ptr = right_adj_vectors.indptr
    rav_col = right_adj_vectors.indices
    cosines = np.zeros(len(rav_ptr) - 1)
    return _cosines(lav_ptr, lav_col, rav_ptr, rav_col, cosines).reshape(-1, 1)

def compute_three_paths(pairs, graph, m):
    """
    Computes number of paths of legth 3 between vertices in pairs.

    PARAMETERS
    ----------
    pairs : np.ndarray(dtype=np.int32, shape=(k, 2))
        array of pairs
    graph : CsrGraph
        graph to use

    RETURNS
    -------
    np.ndarray(dtype=np.int32, shape=(k,1))
        array of number of paths of length 3
    """
    row_ptrs = graph.row_ptrs
    col_indxs = graph.row_indxs
    connectivities = np.zeros((len(pairs), 1))
    u_neighb = m[pairs[:, 1]].todense()
    return _compute_three_paths_aggregate(connectivities, pairs[:, 0],
                                         row_ptrs, col_indxs,
                                         u_neighb)

def ppr_preprocess(graph, alpha=0.8, tol=0.01, workers=20):
    """
    Computes the approximate PPR matrix
    PARAMETERS
    ----------
    alpha : float
        teleportation parameter for PPR, 0 < alpha < 1
    tol : float
        approximation parameter for PPR. Lower tol leads to more memory
        but better approximations, 0 < tol < 1
    num_rows=20 : int
        number of rows to use in training set in addition to the PPR clusters
    workers=20 : int
        number of workers to use for PPR

    RETURNS
    -------
    scipy.sparse.csr_matrix
        approximate PPR matrix
    """

    Gvol = len(graph.row_indxs)
    seeds = range(graph.number_of_nodes())
    p = Pool(workers)
    return sparse.vstack(p.starmap(ppr_cluster, [(graph, Gvol, [seed],
                                                  alpha, tol) for seed in seeds]))

def ppr_cluster(G, Gvol, seed, alpha=0.99, tol=0.0001):
    """G : dictlike adjacency representation of graph
       Gvol : int volume of G
       seed : Iterable of ints, seeds for PPR cluster
       alpha : float < 1, teleportation parameter for PPR
       tol : float < 1, size parameter for PPR

       Return value: a set containing the cluster generated
       from the given parameters

       note: this is stolen from dgleich's github page originally
       """
    if len(G[seed[0]]) == 0:
        return sparse.csr_matrix(([], [], []))
    p = np.zeros(len(G))
    r = np.zeros(len(G))
    Q = collections.deque()  # initialize queue
    r[seed] = 1/len(seed)
    Q.extend(s for s in seed)
    while len(Q) > 0:
        v = Q.popleft()  # v has r[v] > tol*deg(v)
        p, r_prime = push(v, np.copy(r), p, G.row_ptrs, G.row_indxs, alpha)
        new_verts = np.where(r_prime - r > 0)[0]
        r = r_prime
        Q.extend(u for u in new_verts if r[u] / len(G[u]) > tol)
    return sparse.csr_matrix(p)

@njit()
def push(u, r, p, adj_ptrs, adj_cols, alpha):
    r_u = r[u]
    p[u] += alpha * r_u
    r[u] = (1 - alpha) * r_u / 2
    r[adj_cols[adj_ptrs[u]:adj_ptrs[u + 1]]
      ] += (1 - alpha) * r_u / (2 * (adj_ptrs[u + 1] - adj_ptrs[u]))
    return p, r

@njit(parallel=True)
def _compute_three_paths_aggregate(feature_vec, vs, row_ptrs, col_indxs, u_neighb):
    for i in prange(len(vs)):
        v = vs[i]
        for k in col_indxs[row_ptrs[v] : row_ptrs[v + 1]]:
            for l in col_indxs[row_ptrs[k]:row_ptrs[k + 1]]:
                if u_neighb[i, l]:
                    feature_vec[i] += 1
    return feature_vec

@njit(parallel=True)
def _cosines(lav_ptr, lav_col, rav_ptr, rav_col, cosines):
    for i in prange(len(cosines)):
        cosines[i] = _cosine_sim_pair(lav_col[lav_ptr[i]:lav_ptr[i + 1]],
                                      rav_col[rav_ptr[i]:rav_ptr[i + 1]])
    return cosines


@njit()
def _cosine_sim_pair(left_ind, right_ind):
    if len(left_ind) == 0 or len(right_ind) == 0:
        return 0.0
    factor = 1 / np.sqrt(len(left_ind) * len(right_ind))
    cosine = 0
    i = 0
    j = 0
    while i < len(left_ind) and j < len(right_ind):
        if left_ind[i] == right_ind[j]:
            cosine += 1
            i += 1
            j += 1
        elif left_ind[i] < right_ind[j]:
            i += 1
        else:
            j += 1
    return factor * cosine

