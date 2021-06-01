from emBench.scorers.embeddingScorerBase import _EmbeddingScorerBaseClass
from emBench.scorers.fastwalker import do_walks
from gensim.models.word2vec import Word2Vec
from numpy import zeros

class DeepWalkScorer(_EmbeddingScorerBaseClass):
    """
    DeepWalk embedding scorer class

    ATTRIBUTES
    ----------
    _emb : np.ndarray(dtype=np.float32, shape=(n, d)
        matrix representing node embeddings. Each row
        represents the embedding of a vertex in d-dimensions.

    METHODS
    -------
    fit_to_graph(self, graph, walk_number=10, walk_length=80, dimensions=128,
                 workers=4, window_size=5, epochs=1, learning_rate=0.05)
        Fits a DeepWalk embedding to the given graph.
    score_pairs(pairs)
        Returns scores for pairs.
    save(filename)
        Save object as pickle in filename.
    """

    def fit_to_graph(self, graph, walk_number=10, walk_length=80, dimensions=128,
             workers=4, window_size=5, epochs=1, learning_rate=0.05):
        """
        Fits a DeepWalk embedding to the given graph.

        PARAMETERS
        ----------
        graph : CsrGraph
            graph to which to fit an embedding
        walk_number : int, optionl
            number of walks for DeepWalk
        walk_length : int, optionl
            length of walks for DeepWalk
        dimensions : int, optionl
            number of dimensions for the embedding
        workers : int, optionl
            number of workers for the Word2Vec step
            (random walks use all available cores)
        window_size : int, optionl
            window size for Word2Vec
        epochs : int, optionl
            number of iterations for Word2Vec
        learning_rate : float, optionl
            parameter for Word2Vec
        """
        walk_container = do_walks(graph, walk_length, walk_number)
        model = Word2Vec(walk_container,
                         hs=1,
                         alpha=learning_rate,
                         iter=epochs,
                         size=dimensions,
                         window=window_size,
                         min_count=1,
                         workers=workers,
                         )
        emb = zeros((graph.number_of_nodes(), dimensions))
        for i in range(graph.number_of_nodes()):
            emb[i, :] = model[str(i)]
        self._emb = emb
