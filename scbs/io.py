import h5py
import scipy.sparse as sparse


def write_sparse_hdf5(h5object, matrix):
    h5object["indptr"] = matrix.indptr
    h5object["data"] = matrix.data
    h5object["indices"] = matrix.indices
    h5object.attrs["format"] = matrix.format
    h5object["shape"] = matrix.shape


def read_sparse_hdf5(h5object):
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


def iterate_chromosomes(h5object):
    """Read the matrices saved in the groups and return them."""
    groups = h5object.keys()
    for g in groups:
        yield g, read_sparse_hdf5(h5object[g])


def read_chromosome(filename, chrom):
    with h5py.File(filename) as h5object:
        return read_sparse_hdf5(h5object[chrom])


def iterate_file_chromosome(filename):
    with h5py.File(filename, "r") as hfile:
        yield from iterate_chromosomes(hfile)


def write_sparse_hdf5_stream(h5object: h5py.Group, coo_stream, row_num, col_num, nnz):
    """

    We assume that there are no duplicates and the values are ordered by (i, j).

    :param h5object: h5 object
    :param coo_stream: row, col, value
    :param row_num: size of 1st dimension (i.e. chromosome length)
    :param col_num: size of 2nd dimension (i.e. cell number)
    :param nnz: number of data elements

    """
    h5object.attrs["format"] = "csr"
    h5object["shape"] = (row_num, col_num)
    indptr = h5object.create_dataset("indptr", shape=row_num + 1)
    data = h5object.create_dataset("data", shape=nnz)
    indices = h5object.create_dataset("indices", shape=nnz)

    dataidx = 0
    curr_i = 0
    for i, j, value in coo_stream:
        data[dataidx] = value
        indices[dataidx] = j
        if curr_i != i:
            curr_i = i
            indptr[i] = dataidx
        dataidx += 1
    indptr[curr_i + 1] = dataidx

    return h5object
