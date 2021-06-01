from emBench.scorers.pairScorerBase import _PairScorerBaseClass
from emBench.scorers import handTunedFeatures
from emBench.graphs.csrGraph import CsrGraph
from emBench.utils import sample_from_comm
import numpy as np
from sklearn.linear_model import LogisticRegression

class HandTunedScorer(_PairScorerBaseClass):
    """
    Scorer class for hand-selected graph features.

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

    def score_pairs(self, pairs, chunksize=1000):
        """returns feature vectors for pairs

        PARAMETERS
        ----------
        pairs : np.ndarray(dtype=np.int32, shape=(k, 2))
            array with rows representing pairs of vertices in graph
        chunksize : int
            number of pairs to compute scores for at once, controls memory footprint

        RETURNS
        -------
        np.darray(dtype=np.float32, shape=(k, d))
            array of scores
        """
        pair_features = self.pair_features(pairs, chunksize=chunksize)
        return self._model.predict_proba(pair_features)[:, 1].reshape(-1, 1)

    def fit_to_graph(self, graph, alpha=0.8, tol=.001, num_rows=20, workers=20, ppr=None):
        """
        fits a logistic regression model to the graph and stores it

        PARAMETERS
        ----------
        graph : CsrGraph
            graph to which a model will be fit
        alpha : float
            teleportation parameter for PPR, 0 < alpha < 1
        tol : float
            approximation parameter for PPR. Lower tol leads to more memory
            but better approximations, 0 < tol < 1
        num_rows=20 : int
            number of rows to use in training set in addition to the PPR clusters
        workers=20 : int
            number of workers to use for PPR
        ppr=None : scipy.sparse.csr_matrix
            optional parameter to skip ppr step
        """
        self._adj = CsrGraph(graph)
        if ppr is None:
            self._ppr = handTunedFeatures.ppr_preprocess(self._adj, alpha, tol, workers)
        else:
            self._ppr = ppr
        training_set = self._get_training_pairs(num_rows)
        labels = graph.label_pairs(training_set)
        features = self.pair_features(training_set)
        self._model = LogisticRegression(max_iter=10000).fit(features, labels)

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
        # to_csr_matrix gets expensive for large graphs for some reason
        # I think, so we just make it once (would be more elegant
        # to put it in the batch function body)
        m = self._adj.to_csr_matrix().tolil()
        chunked_pairs = [pairs[i * chunksize:(i + 1) * chunksize]
                         for i in range(int(len(pairs) / chunksize))]
        if int(len(pairs) / chunksize) * chunksize != len(pairs):
            chunked_pairs += [pairs[int(len(pairs) / chunksize) * chunksize:]]
        return np.vstack([self._batch_features(batch, m) for batch in chunked_pairs])

    def _batch_features(self, pairs, m):
        if len(pairs) == 0:
            return
        cosines = handTunedFeatures.cosine_sim(pairs, self._adj.to_csr_matrix())
        connectivities = handTunedFeatures.compute_three_paths(pairs, self._adj, m)

        pprs = np.array(list(map(
            lambda x: [self._ppr[x[0], x[1]], self._ppr[x[1], x[0]]], pairs)))
        return np.hstack((cosines, connectivities, pprs))

    def _get_training_pairs(self, num_rows):
        ppr_pairs = np.vstack(self._ppr.nonzero()).T
        row_pairs = np.vstack([np.array([[u, v] for v in self._adj.nodes()])
                               for u in np.random.randint(0, self._adj.number_of_nodes(), num_rows, np.int32)])
        return np.vstack((ppr_pairs, row_pairs))
