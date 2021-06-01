from emBench.scorers.handTunedScorer import HandTunedScorer
from emBench.scorers import handTunedFeatures
from emBench.graphs.csrGraph import CsrGraph
from emBench.utils import sample_from_comm
import numpy as np
from sklearn.linear_model import LogisticRegression

class HadamardScorer(HandTunedScorer):
    """
    Scorer class for hadamard products of embedding vectors.

    METHODS
    -------
    score_pairs(pairs)
        returns scores for pairs
    save(filename)
        save object as pickle in filename
    fit_to_graph(self, graph, alpha=0.8, tol=.001, num_rows=20, workers=20)
        fits a model to the graph and stores it
    pair_features(pairs, chunksize=40000)
        returns feature vectors for pairs
    """

    def __init__(self, emb):
        """
        emb is the embedding to train a Hadamard product
        scorer on.
        """
        self._emb = emb

    def pair_features(self, pairs, chunksize=10000):
        """Returns feature arrays for each pair in pairs

        PARAMETERS
        ----------
        pairs : np.ndarray(dtype=np.int32, shape=(k, 2))
            array with rows representing pairs of vertices in graph
        chunksize=40000 : int
            number of pairs to compute at once, controls memory footprint


        RETURNS
        -------
        np.ndarray(dtype=np.float32, shape=(k, 3))
            feature vectors for pairs
        """
        return np.multiply(self._emb[pairs[:, 0]], self._emb[pairs[:, 1]])
