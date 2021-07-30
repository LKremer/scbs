import h5py
import scipy.sparse as sparse


def write_sparse_hdf5(h5object: h5py.AttributeManager, matrix):
    h5object["indptr"] = matrix.indptr
    h5object["data"] = matrix.data
    h5object["indices"] = matrix.indices
    h5object["format"] = matrix.format


def read_sparse_hdf5(h5object: h5py.AttributeManager):
    """Read the matrix from the provided h5py File or Group."""
    try:
        matformat = h5object["format"]
        constructor = {"csc": sparse.csc_matrix, "csr": sparse.csr_matrix}[
            matformat
        ]
    except KeyError:
        raise KeyError(
            "The matrix format (csc, csr or coo) must be specified in the 'format' attribute."
        )
    try:
        return constructor(
            h5object["data"], h5object["indices"], h5object["indptr"]
        )
    except KeyError:
        raise Exception(
            """There must be provided three arrays for the compressed format: \
            'data', 'indices' and 'indptr'."""
        )
