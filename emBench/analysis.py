"""Tools for analyzing the performance of graph embeddings.

METHODS
-------
pr_curve(graph, scorer, num_rows)
    Computes an approximate precision-recall curve.
median_p_at_k_curve(labeledGraph, scorer, num_rows, k, workers=72)
    Returns the median precision over num_rows rows when the classifier
    is thresholded for the top i for all i in range(k)
"""
from multiprocessing import Pool
from sklearn.metrics import precision_score, precision_recall_curve
from emBench.utils import sample_rows
import numpy as np


def pr_curve(graph, scorer, num_rows):
    """Computes an approximate precision-recall curve.

    PARAMETERS
    ----------
    graph : emBench.graphs.LabeledGraph
        The graph with pair labels for evaluation
    scorer : emBench.scorers._PairScorerBase
        The scorer object we wish to evaluate
    num_rows : int
        number of rows to use to estimate

    RETURNS
    _______
    (r, p) : (np.ndarray, np.ndarray)
        arrays containing recalls and precisions respectively at each
        threshold
    """
    pairs = sample_rows(graph, num_rows)
    labels = graph.label_pairs(pairs)
    features = scorer.score_pairs(pairs)
    p, r, _ = precision_recall_curve(labels, features)
    return r, p

def median_p_at_k_curve(labeledGraph, scorer, num_rows, k, workers=72):
    """Returns the median precision over num_rows rows when the classifier
    is thresholded for the top i for all i in range(k)

    PARAMETERS
    ----------
    labeledGraph : emBench.graphs.LabeledGraph
        The graph with pair labels for evaluation
    scorer : emBench.scorers._PairScorerBase
        The scorer object we wish to evaluate
    num_rows : int
        number of rows to use to estimate
    k : int
        top cutoff value to go up to
    workers : int
        number of threads to use

    RETURNS
    -------
    numpy ndarray of floats containing the median precisions with shape (k, ).
    The entry at position[i] represents the median precision with thresholding at the top i.
    """

    ps = np.zeros((num_rows, k))
    p = Pool(workers)
    for i in range(num_rows):
        pairs = sample_rows(labeledGraph, 1)
        labels = labeledGraph.label_pairs(pairs)
        feats = scorer.score_pairs(pairs)
        x = np.argsort(-feats, axis=0)
        new_ps = p.starmap(_top_ps, [(feats, labels, x, i) for i in range(1, k + 1, 1)])
        ps[i] = new_ps
    return np.median(ps, axis=0)

def _top_ps(scores, labels, x, k):
    """Computes the top k precision of the given samples.

    PARAMETERS
    ----------
    scores : numpy.ndarray
        array of pair scores with shape (n, 1)
    labels : numpy.ndarray
        labels of pairs with indices corresponding to pairs in scores
    x : numpy.ndarray
        a copy of scores but sorted

    RETURNS
    -------
    float containing the precision score when the top k pairs are
       predicted as True, and the rest are False

    """
    # TODO: arguments are weird maybe bc I was trying to use numba? 
    # the easiest thing is just to sort labels too and then pass it in
    y_pred = np.zeros(len(scores), dtype=bool)
    y_pred[x[:k]] = 1
    return precision_score(labels, y_pred)

def max_p_at_k_curve(labeledGraph, scorer, num_rows, k, workers=72):
    """Returns the median precision over num_rows rows when the classifier
    is thresholded for the top i for all i in range(k)

    PARAMETERS
    ----------
    labeledGraph : emBench.graphs.LabeledGraph
        The graph with pair labels for evaluation
    scorer : emBench.scorers._PairScorerBase
        The scorer object we wish to evaluate
    num_rows : int
        number of rows to use to estimate
    k : int
        top cutoff value to go up to
    workers : int
        number of threads to use

    RETURNS
    -------
    numpy ndarray of floats containing the median precisions with shape (k, ).
    The entry at position[i] represents the median precision with thresholding at the top i.
    """

    ps = np.zeros((num_rows, k))
    p = Pool(workers)
    for i in range(num_rows):
        pairs = sample_rows(labeledGraph, 1)
        labels = labeledGraph.label_pairs(pairs)
        feats = scorer.score_pairs(pairs)
        x = np.argsort(-feats, axis=0)
        new_ps = p.starmap(_top_ps, [(feats, labels, x, i) for i in range(1, k + 1, 1)])
        ps[i] = new_ps
    return np.max(ps, axis=0)

