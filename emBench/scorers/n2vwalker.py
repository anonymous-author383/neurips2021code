"""Module to do walks for Node2Vec

METHODS
-------
do_walks(graph, walk_length, walk_number, p, q, workers=1)
    performs walks and returns iterator over walk transcripts
"""
import numpy as np
from numba import njit, prange
from emBench.scorers.fastwalker import _WalkContainer, process_graph


@njit()
def _do_step(t, v, p, q, t_neighbs, v_neighbs):
    # first we need to calculate the distribution over possible
    # next steps
    index = 0
    domain = np.concatenate((t_neighbs, v_neighbs)).astype(np.int32)
    ps = np.concatenate((t_neighbs, v_neighbs)).astype(np.float32)
    for u in v_neighbs:
        domain[index] = u
        # I would like to check if u in t_neighbs, but numba wont
        # let me use "in" here, have to write our own search
        # TODO: would binary search be worth implementing here?
        in_t_flag = False
        for w in t_neighbs:
            if u == w:
                in_t_flag = True
                break
        if in_t_flag:
            ps[index] = 1.0
        elif u == t:
            ps[index] = 1 / p
        else:
            ps[index] = 1 / q
        index += 1
    ps[index:] = 0
    # normalize and make ps contain the cdf
    mass = 0
    ps = ps / np.sum(ps)
    for i in range(index):
        mass += ps[i]
        ps[i] = mass
    # now sample from distribution and return endpoint
    return domain[_prob_map(np.random.random_sample(), ps, 0, index)]


def _do_n2v_walks(p, q, length, rows, indxs, chunksize=10000):
    """Breaks walks up into chunks for memory reasons and performs the walks on them"""
    n = len(indxs) - 1
    transcripts = np.zeros((0, length), dtype=np.int32)
    for chunk_start in range(0, n, chunksize):
        chunk_stop = min(chunk_start + chunksize, n)
        # create the arrays for this batch
        t_neighbs_chunk = np.zeros((chunk_stop - chunk_start, n), dtype=np.bool_)
        v_neighbs_chunk = np.zeros((chunk_stop - chunk_start, n), dtype=np.bool_)
        transcript_arr = np.zeros((chunk_stop - chunk_start, length), dtype= np.int32)
        # since Node2Vec walks depend on the last two vertices, we need to start the
        # the first two columns of the transcript array to contain the walk startpoints
        transcript_arr[:, 0] = range(chunk_start, chunk_stop)
        transcript_arr[:, 1] = range(chunk_start, chunk_stop)
        # do the walks
        transcript_arr = _batch_walks(transcript_arr, length, p, q,
                                 rows, indxs, t_neighbs_chunk, v_neighbs_chunk)
        transcripts = np.vstack((transcripts, transcript_arr))
    return transcripts

@njit(parallel=True)
def _batch_walks(transcript_arr, length, p, q, rows, indxs, t_neighbs_chunk, v_neighbs_chunk):
    """Performs walks for all vertices in the first two columns of transcript_arr"""
    # walks for each vertex are done in parallel
    for i in prange(transcript_arr.shape[0]):
        for step in range(length):
            # as far as I can tell, I can't do prange(start, stop), so this
            # continue is necessary.
            if step < 2:
                continue
            v = transcript_arr[i, step - 1]
            t = transcript_arr[i, step - 2]
            t_neighbs = rows[indxs[v]:indxs[v + 1]]
            v_neighbs = rows[indxs[t]:indxs[t + 1]]
            # get the probability distribution over the next step
            transcript_arr[i, step] = _do_step(t, v, p, q, t_neighbs, v_neighbs)
    return transcript_arr

@njit()
def _prob_map(val, p_arr, beg, end):
    """returns the least index into p_arr with value at least val"""
    # binary search, assumes p_arr is sorted
    if end - beg <= 1:
        return beg
    else:
        pivot = beg + ((end - beg) // 2)
        if val < p_arr[pivot]:
            return _prob_map(val, p_arr, beg, pivot)
        else:
            return _prob_map(val, p_arr, pivot, end)


def do_walks(graph, walk_length, walk_number, p, q, chunksize=4000):
    """Perform node2vec's second order walks

    PARAMETERS
    ----------
    graph : CsrGraph
        graph to do walks on
    walk_length : int
        length of random walks
    walk_number : int
        number of random walks per vertex
    p : float
        parameter for Node2Vec walks
    q : float
        parameter for Node2Vec walks
    chunksize=4000 : int
        number of walks to compute at once. Controls memory usage

    RETURNS
    -------
    iterator containing walk transcripts as lists of strings
    """
    rows, indxs = process_graph(graph)
    n = len(indxs) - 1
    # pool = Pool(processes=workers)
    # args = [(x, walk_length, p, q, rows, indxs)
    #         for x in range(n) for _ in range(walk_number)]
    np.random.seed()
    walk_container = _WalkContainer(n * walk_number, walk_length)
    walk_container.walks = np.vstack([_do_n2v_walks(
        p, q, walk_length, rows, indxs, chunksize) for _ in range(walk_number)])
    return walk_container
