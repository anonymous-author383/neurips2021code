from emBench.scorers.n2vwalker import do_walks
from emBench.scorers.embeddingScorerBase import _EmbeddingScorerBaseClass
from gensim.models.word2vec import Word2Vec
import numpy as np


class Node2VecScorer(_EmbeddingScorerBaseClass):
    """
    Node2Vec embedding scorer class

    ATTRIBUTES
    ----------
    _emb : np.ndarray(dtype=np.float32, shape=(n, d))
        matrix representing node embeddings. Each row
        represents the embedding of a vertex in d-dimensions.

    METHODS
    -------
    fit_to_graph(self, graph, walk_number=10, walk_length=80, dimensions=128,
                 workers=4, window_size=5, epochs=1, learning_rate=0.05)
        Fits a Node2Vec embedding to the graph.
    score_pairs(pairs)
        Returns dot product of embeddings for pairs.
    save(filename)
        Save object as pickle in filename.
    """

    def fit_to_graph(self, graph, walk_number=10, walk_length=80, dimensions=128,
                     workers=4, window_size=5, epochs=1, learning_rate=0.05, p=0.5, q=0.5):
        """Fits a Node2Vec embedding to the given graph

        PARAMETERS
        ----------
        graph : CsrGraph
            graph to which to fit an embedding
        walk_number : int, optional
            number of walks for Node2Vec
        walk_length : int, optional
            length of walks for Node2Vec
        dimensions : int, optional
            number of dimensions for the embedding
        workers : int, optional
            number of workers for the Word2Vec step, default is 4.
        window_size : int, optional
            window size for Word2Vec, default is 5
        epochs : int, optonal
            number of iterations for Word2Vec, default is 1
        learning_rate=0.05
            parameter for Word2Vec
        p : float, optional
            parameter for Node2Vec walks
        q : float, optional
            parameter for Node2Vec walks
        """
        walk_container = do_walks(graph, walk_length, walk_number, p, q)
        model = Word2Vec(walk_container,
                         hs=1,
                         alpha=learning_rate,
                         iter=epochs,
                         size=dimensions,
                         window=window_size,
                         min_count=1,
                         workers=workers)
        emb = np.zeros((graph.number_of_nodes(), dimensions))
        for i in range(graph.number_of_nodes()):
            emb[i, :] = model[str(i)]
        self._emb = emb
