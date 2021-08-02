from scbs.io import (
    write_sparse_hdf5,
    read_sparse_hdf5,
    iterate_chromosomes,
    read_chromosome,
)
import scipy.sparse as sparse
import pytest
import h5py


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
