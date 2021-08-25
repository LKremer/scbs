import scbs.iosparse as sios
import pytest
from scbs.io import write_sparse_hdf5
import scipy.sparse as sparse
import h5py
import numpy as np


sparse_mats = [
    sparse.csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
    sparse.csr_matrix([[1, 1, 0], [2, 2, 0], [0, 0, 0]]),
    sparse.csr_matrix([[1, 1, 0], [2, 2, 0], [0, 0, 0], [0, 0, 0]]),
]


@pytest.mark.parametrize("matrix", sparse_mats)
def test_access(tmpdir, matrix):
    p = tmpdir
    hfile = p / "test.hdf5"
    with h5py.File(hfile, "w") as h5object:
        write_sparse_hdf5(h5object, matrix)
    with h5py.File(hfile) as h5object:
        sparseFacade = sios.CreateSparseHDF5Facade(h5object)
        data = np.array(sparseFacade.data)
        indices = np.array(sparseFacade.indices)
        indptr = np.array(sparseFacade.indptr)

        np.testing.assert_equal(matrix.data, data)
        np.testing.assert_equal(matrix.indices, indices)
        np.testing.assert_equal(matrix.indptr, indptr)
