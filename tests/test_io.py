from scbs.io import (
    write_sparse_hdf5,
    read_sparse_hdf5,
    iterate_chromosomes,
    read_chromosome,
    write_sparse_hdf5_stream,
)
import scipy.sparse as sparse
import pytest
import h5py
import numpy as np


@pytest.fixture
def matrices():
    a = sparse.csr_matrix([[1, 1], [2, 2]])
    b = sparse.csr_matrix([[1, 1], [3, 3]])
    c = sparse.csr_matrix([[1, 1], [4, 4]])
    return [a, b, c]


def test_read_write(matrices, tmpdir):
    p = tmpdir
    hfile = p / "test.hdf5"
    with h5py.File(hfile, "w") as h5object:
        write_sparse_hdf5(h5object, matrices[0])
    with h5py.File(hfile, "r") as h5object:
        a = read_sparse_hdf5(h5object)
        assert (a != matrices[0]).nnz == 0


def test_iterate_chromosomes(matrices, tmpdir):
    p = tmpdir
    hfile = p / "test.hdf5"
    with h5py.File(hfile, "w") as h5object:
        for i, m in enumerate(matrices):
            group = h5object.create_group(str(i))
            write_sparse_hdf5(group, m)
    with h5py.File(hfile, "r") as h5object:
        result = iterate_chromosomes(h5object)
        for i, m in enumerate(matrices):
            test_i, test_m = next(result)
            assert str(i) == test_i
            assert m.format == test_m.format
            assert (m != test_m).nnz == 0


def test_read_chromosomes(matrices, tmpdir):
    p = tmpdir
    hfile = p / "test.hdf5"
    with h5py.File(hfile, "w") as h5object:
        for i, m in enumerate(matrices):
            group = h5object.create_group(str(i))
            write_sparse_hdf5(group, m)
    for i, m in enumerate(matrices):
        test_m = read_chromosome(hfile, str(i))
        assert m.format == test_m.format
        assert (m != test_m).nnz == 0


def iter_coo(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j]:
                yield i, j, x[i, j]


def swap_row_col(coo_stream):
    for i, j, v in coo_stream:
        yield j, i, v


sparse_mats = [
    sparse.csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
    sparse.csr_matrix([[1, 1, 0], [2, 2, 0], [0, 3, 3]]),
]


@pytest.mark.parametrize("matrix", sparse_mats)
def test_stream_write_CSR(matrix, tmpdir):
    p = tmpdir
    hfile = p / "test.hdf5"
    with h5py.File(hfile, "w") as h5object:
        coo_stream = iter_coo(matrix)
        row_num, col_num = matrix.shape
        nnz = matrix.getnnz()
        write_sparse_hdf5_stream(h5object, coo_stream, row_num, col_num, nnz)
        data, indices, indptr = (
            np.array(h5object["data"]),
            np.array(h5object["indices"]),
            np.array(h5object["indptr"]),
        )
        np.testing.assert_equal(matrix.data, data)
        np.testing.assert_equal(matrix.indices, indices)
        np.testing.assert_equal(matrix.indptr, indptr)
