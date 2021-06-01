from abc import abstractclassmethod, ABC
import pickle as pkl

class _PairScorerBaseClass(ABC):
    """Abstract class which requires subclasses to implement score_pairs.
    This is meant to represent the various pair scoring methods.

    METHODS
    -------
    score_pairs(pairs)
        returns scores for pairs
    save(filename)
        save object as pickle in filename
    """

    @abstractclassmethod
    def score_pairs(self, pairs, **kwargs):
        """returns scores for pairs

        PARAMETERS
        ----------
        pairs : np.ndarray(dtype=np.int32, shape=(k, 2))
            array with rows representing pairs of vertices in graph

        RETURNS
        -------
        np.ndarray(dtype=np.float32, shape=(k, )
            array of scores corresponding to pairs
        """
        pass

    def save(self, filename):
        """Saves object as a pickle file.

        PARAMETERS
        ----------
        filename: String
            path to save file in
        """
        filep = open(filename, "wb")
        pkl.dump(self, filep)
        filep.close()
