from abc import abstractclassmethod
from emBench.scorers.pairScorerBase import _PairScorerBaseClass
import numpy as np

class _EmbeddingScorerBaseClass(_PairScorerBaseClass):
    """Abstract base class for the various embedding methods,
    stores an embedding and scores via dot products

    ATTRIBUTES
    ----------
    _emb : np.ndarray(dtype=np.float32, shape=(n, d)
        matrix representing node embeddings. Each row
        represents the embedding of a vertex in d-dimensions.

    METHODS
    -------
    score_pairs(pairs)
        Returns scores for pairs
    save(filename)
        Save object as pickle in filename
    """

    def score_pairs(self, pairs, chunksize=10000):
        """Returns dot product of embedding vectors as scores.
        pairs : np.ndarray(dtype=np.int32, shape=(k, 2))
            Each row is a pair of node ids.
        """
        n = len(pairs)
        scores = np.zeros((0, 1), dtype=np.float32)
        for chunk_start in range(0, n, chunksize):
            chunk_stop = min(chunk_start + chunksize, n)
            pair_products = self._emb[pairs[chunk_start:chunk_stop, 0], :] * self._emb[pairs[chunk_start:chunk_stop, 1], :]
            chunk_scores = np.sum(pair_products, axis=1).reshape(-1, 1)
            scores = np.vstack((scores, chunk_scores))
        return scores
