"""module for the CsrDict data structure"""
import numpy as np
from collections.abc import Mapping
from scipy.sparse import csr_matrix


class CsrDict(Mapping):
    """Class to provide efficient *immutable* maps for sparse datasets.
    exposes a dictlike interface, but errors out on any attempt to set items

    Every CsrDict maps all keys in range(n) to an np.ndarray of np.int32's

    CSR stands for "compressed sparse row" - see the wikipedia link for details.
    Here, we assume all the entries are just ones, and do not need the data array.
    https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)


    ATTRIBUTES
    ----------
    row_indxs : numpy.ndarray
        array containing the column ids for all the values stored
    row_ptrs : nump.ndarray
        array containing the pointers for new row locations into row_indxs

    METHODS
    -------
    keys()
        returns all keys in CsrDict
    to_csr_matrix()
        returns a version of the CsrDict as a scipy.sparse.csr_matrix
    transpose()
        returns a CsrDict keyed by values of self and with values equal
        to the set of keys in self which map to a given index
    """

    def __init__(self, arg1, symmetric=False):
        """
        PARAMETERS
        ----------
        arg1 : list of dicts OR scipy.csr_matrix OR dictlike
            the object to convert to CsrDict

        symmetric=False : bool
            flag for whether or not to make sure symmetric entries are
            added to make an undirected graph. Default False since it comes
            with performance penalties
        """
        if isinstance(arg1, list):
            CsrDict.__init__(self, {i: c for i, c in enumerate(arg1)})
        elif isinstance(arg1, csr_matrix):
            self.row_indxs = arg1.indices
            self.row_ptrs = arg1.indptr
        else:
            dictlike = arg1
            row_lens = [len(list(row)) for row in dictlike.values()]
            self.row_indxs = np.zeros(int(np.sum(row_lens)), dtype=np.int32)
            self.row_ptrs = np.zeros(len(dictlike) + 1, dtype=np.int32)
            index = 0
            for v in range(len(dictlike)):
                row = list(dictlike[v])
                self.row_indxs[index:index + len(row)] = row
                self.row_ptrs[v] = index
                index += len(row)
            self.row_ptrs[-1] = index

        if symmetric:
            new_dictlike = dict(self)
            for key, vals in self.items():
                for val in vals:
                    if key not in self[val]:
                        new_dictlike[val] = sorted([key] + new_dictlike.get(val, []))
            CsrDict.__init__(self, new_dictlike, symmetric=False)


    def __getitem__(self, key):
        """Returns the array of values corresponding to key

        PARAMETERS
        ----------
        key : int
            a key in the dict

        RETURNS
        -------
        np.ndarray of values
        """
        # TODO should I be raising a value error?
        if key == len(self.row_ptrs) - 1:
            return self.row_indxs[self.row_ptrs[key]:]
        elif key >= len(self.row_ptrs):
            return []
        return self.row_indxs[self.row_ptrs[key]:self.row_ptrs[key + 1]]

    def keys(self):
        """Returns the keys in this object

        RETURNS
        -------
        np.ndarray(dtype=np.intew)
        """
        return np.arange(len(self), dtype=np.int32)

    def __len__(self):
        return len(self.row_ptrs) - 1

    def __iter__(self):
        for i in range(len(self)):
            yield i

    def to_csr_matrix(self):
        """Convert self to scipy.csr_matrix

        RETURNS
        _______
        scipy.sparse.csr_matrix
        """
        return csr_matrix((np.ones(len(self.row_indxs), dtype=np.bool_),
                           self.row_indxs, self.row_ptrs))

    def transpose(self):
        """Returns a CsrDict keyed by values of self and with values equal
        to the set of keys in self which map to a given index

        RETURNS
        -------
        CsrDict
        """
        a = self.to_csr_matrix()
        b = a.tocsc()
        c = CsrDict(dict())
        c.row_ptrs = b.indptr
        c.row_indxs = b.indices
        return c
