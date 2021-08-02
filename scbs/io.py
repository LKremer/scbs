import h5py
import scipy.sparse as sparse


def write_sparse_hdf5(h5object: h5py.AttributeManager, matrix):
    h5object["indptr"] = matrix.indptr
    h5object["data"] = matrix.data
    h5object["indices"] = matrix.indices
    h5object.attrs["format"] = matrix.format
    h5object["shape"] = matrix.shape


def read_sparse_hdf5(h5object: h5py.AttributeManager):
    """Read the matrix from the provided h5py File or Group."""
    try:
        constructor = {"csc": sparse.csc_matrix, "csr": sparse.csr_matrix}[
            h5object.attrs["format"]
        ]
    except KeyError:
        raise KeyError(
            "The matrix format (csc, csr or coo) must be specified in the 'format' attribute."
        )
    try:
        return constructor(
            (h5object["data"], h5object["indices"], h5object["indptr"]),
            shape=h5object["shape"],
        )
    except KeyError:
        raise Exception(
            """There must be provided three arrays for the compressed format: \
            'data', 'indices' and 'indptr'."""
        )


def iterate_chromosomes(h5object: h5py.AttributeManager):
    """Read the matrices saved in the groups and return them."""
    groups = h5object.keys()
    for g in groups:
        yield g, read_sparse_hdf5(h5object[g])


def read_chromosome(filename, chrom):
    with h5py.File(filename) as h5object:
        return read_sparse_hdf5(h5object[chrom])
