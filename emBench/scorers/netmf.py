from emBench.scorers._netmfUtils import _create_target_matrix, _create_embedding
from emBench.scorers.embeddingScorerBase import _EmbeddingScorerBaseClass

class NetMFScorer(_EmbeddingScorerBaseClass):
    """
    NetMF embedding scorer class

    ATTRIBUTES
    ----------
    _emb : np.ndarray(dtype=np.float32, shape=(n, d)
        matrix representing node embeddings. Each row
        represents the embedding of a vertex in d-dimensions.

    METHODS
    -------
    fit_to_graph(self, graph, walk_number=10, walk_length=80, dimensions=128,
                 workers=4, window_size=5, epochs=1, learning_rate=0.05)
        Fits a NetMF embedding to the given graph.
    score_pairs(pairs)
        returns scores for pairs
    save(filename)
        save object as pickle in filename
    """

    def fit_to_graph(self, graph, dimensions=128, iterations=10, order=2, negative_samples=1):
        """
        Fits a NetMF embedding to the given graph.

        PARAMETERS
        ----------
        graph : CsrGraph
            Graph to which to fit an embedding
        dimensions : int, optional
            Number of dimensions for embedding, default is 128
        iterations : int, optional
            Number of iterations to run NetMF
        order : int, optional
            Power of matrix to go up to for NetMF
        negative_samples : int, optional
            Parameter for NetMF

        """
        target_matrix = _create_target_matrix(graph, order, negative_samples)
        self._emb = _create_embedding(target_matrix, dimensions, iterations)
