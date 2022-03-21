import h5py
import scipy.sparse as sparse


def CreateSparseHDF5Facade(h5object):
    """Create a duck-typed sparse csr matrix, where inptr,
       indices and data are from HDF5 file.

    :param h5object: Group or File object from h5py
    :returns:

    """
    x = sparse.csr_matrix(tuple(h5object["shape"]))
    x.indices = h5object["indices"]
    x.indptr = h5object["indptr"]
    x.data = h5object["data"]
    return x
