import numpy as np
import h5py
import gzip
import os
import scipy.sparse as sp_sparse
from .utils import echo, secho
import sys
import pandas as pd
from scbs.io import write_sparse_hdf5, write_sparse_hdf5_stream, ChromosomeDataDesc
from collections import Counter
from contextlib import ExitStack
from .coverage_format import create_custom_format, create_standard_format


def prepare(input_files, data_dir, input_format, streamed_write=False):
    cell_names = _get_cell_names(input_files)
    n_cells = len(cell_names)
    # we use this opportunity to count some basic summary stats
    n_obs_cell = np.zeros(n_cells, dtype=np.int64)
    n_meth_cell = np.zeros(n_cells, dtype=np.int64)

    # For each chromosome, we first make a sparse matrix in COO (coordinate)
    # format, because COO can be constructed value by value, without knowing the
    # dimensions beforehand. This means we can construct it cell by cell.
    # We dump the COO to hard disk to save RAM and then later convert each COO to a
    # more efficient format (CSR).
    echo(f"Processing {n_cells} methylation files...")
    coo_files, chrom_descriptions = _dump_coo_files(
        input_files, input_format, n_cells, data_dir
    )
    echo(
        "\nStoring methylation data in 'compressed "
        "sparse row' (CSR) matrix format for future use."
    )
    # read each COO file and convert the matrix to CSR format.
    # Write the matrices to the corresponding groups in the hdf5 file.
    save_method = (
        save_chromosome_compressed_stream
        if streamed_write
        else save_chromosome_compressed
    )
    save_coo_to_compressed(
        coo_files, os.path.join(data_dir, "scbs.hdf5"), chrom_descriptions, save_method
    )

    colname_path = _write_column_names(data_dir, cell_names)
    echo(f"\nWrote matrix column names to {colname_path}")
    stats_path = _write_summary_stats(data_dir, cell_names, n_obs_cell, n_meth_cell)
    echo(f"Wrote summary stats for each cell to {stats_path}")
    secho(
        f"\nSuccessfully stored methylation data for {n_cells} cells "
        f"with {len(coo_files.keys())} chromosomes.",
        fg="green",
    )


def save_coo_to_compressed(coo_files, destination, chrom_descriptions, save_method):
    """Saves all chromosome COO files into a single file with compressed sparse matrix format.

    :param coo_files: a list of file names
    :param destination: file name
    :param chrom_descriptions: list of ChromosomeDataDesc objects (we need to know e.g. the dimensions)
    :param save_method: whether to load all in memory (save_chromosome_compressed) or stream,
      in case it is preferred.

    """
    with h5py.File(destination, "w") as hfile:
        for chrom in coo_files.keys():
            h5object = hfile.create_group(chrom)
            echo(
                f"Reading {chrom_descriptions[chrom].size} x \
                {chrom_descriptions[chrom].n_cells} COO matrix for chromosome {chrom}..."
            )
            echo(f"Writing  {chrom} ...")
            save_method(coo_files[chrom], h5object, chrom_descriptions[chrom])
            os.remove(coo_files[chrom])


def iterate_coo(coo_filename):
    with open(coo_filename) as f:
        for line in f:
            pos, cell, value = line.strip().split(",")
            yield int(pos), int(cell), float(value)


def save_chromosome_compressed(coo_filename, h5object, chrom_desc: ChromosomeDataDesc):
    mat = _load_csr_from_coo(coo_filename, chrom_desc.size, chrom_desc.n_cells)
    write_sparse_hdf5(h5object, mat.tocsr())


def save_chromosome_compressed_stream(coo_filename, h5object, chrom_desc):
    coo_stream = iterate_coo(coo_filename)
    write_sparse_hdf5_stream(
        h5object, coo_stream, chrom_desc.size, chrom_desc.n_cells, chrom_desc.nnz
    )


def _get_cell_names(cov_files):
    """Use the file base names (without extension) as cell names."""
    names = []
    for file_handle in cov_files:
        f = os.path.basename(file_handle.name)
        if f.lower().endswith(".gz"):
            # remove .gz
            f = f[:-3]
        # remove .xxx
        names.append(os.path.splitext(f)[0])
    if len(set(names)) < len(names):
        s = (
            "\n".join(names) + "\nThese sample names are not unique, "
            "check your file names again!"
        )
        raise Exception(s)
    return names


def _dump_coo_files(fpaths, input_format, n_cells, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        c_col, p_col, m_col, u_col, coverage, sep, header = _human_to_computer(
            input_format
        )
    except Exception as e:
        raise type(e)(
            f"{e}\n\nUnknown input file format '{input_format}'.\nValid options "
            "include 'bismark', 'allc', 'methylpy' or a custom ':'-separated format "
            "(check 'scbs prepare --help' for details)."
        ).with_traceback(sys.exc_info()[2])

    coo_files = {}
    chrom_sizes = {}
    chrom_nnz = Counter()
    with ExitStack() as stack:
        for cell_n, cov_file in enumerate(fpaths):
            if cell_n % 50 == 0:
                echo("{0:.2f}% done...".format(100 * cell_n / n_cells))
            for line_vals in _iterate_covfile(
                cov_file, c_col, p_col, m_col, u_col, coverage, sep, header
            ):
                chrom, genomic_pos, n_meth, n_unmeth = line_vals
                if n_meth != 0 and n_unmeth != 0:
                    continue  # currently we ignore all CpGs that are not "clear"!
                meth_value = 1 if n_meth > 0 else -1
                if chrom not in coo_files:
                    coo_path = os.path.join(output_dir, f"{chrom}.coo")
                    coo_files[chrom] = stack.enter_context(open(coo_path, "w"))
                    chrom_sizes[chrom] = 0
                if genomic_pos > chrom_sizes[chrom]:
                    chrom_sizes[chrom] = genomic_pos
                coo_files[chrom].write(f"{genomic_pos},{cell_n},{meth_value}\n")
                chrom_nnz[chrom] += 1
    echo("100% done.")
    coo_filenames = {chrom: coo_files[chrom].name for chrom in coo_files}
    chrom_descriptions = {
        chrom: ChromosomeDataDesc(chrom_sizes[chrom], n_cells, chrom_nnz[chrom])
        for chrom in chrom_sizes
    }
    return coo_filenames, chrom_descriptions


def _load_csr_from_coo(coo_path, chrom_size, n_cells):
    try:
        coo = np.loadtxt(coo_path, delimiter=",", ndmin=2)
        mat = sp_sparse.coo_matrix(
            (coo[:, 2], (coo[:, 0], coo[:, 1])),
            shape=(chrom_size + 1, n_cells),
            dtype=np.int8,
        )
        echo("Converting from COO to CSR...")
        mat = mat.tocsr()  # convert from COO to CSR format
    except Exception as e:
        raise type(e)(f"{e} (problematic file: {coo_path})").with_traceback(
            sys.exc_info()[2]
        )
    return mat


def _write_column_names(output_dir, cell_names, fname="column_header.txt"):
    """
    The column names (usually cell names) will be
    written to a separate text file
    """
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as col_head:
        for cell_name in cell_names:
            col_head.write(cell_name + "\n")
    return out_path


def _human_to_computer(file_format):
    """Convert the human-readable input file format to a tuple."""
    if ":" in file_format:
        return create_custom_format(file_format).totuple()
    return create_standard_format(file_format).totuple()


def _write_summary_stats(data_dir, cell_names, n_obs, n_meth):
    stats_df = pd.DataFrame(
        {
            "cell_name": cell_names,
            "n_obs": n_obs,
            "n_meth": n_meth,
            "global_meth_frac": np.divide(n_meth, n_obs),
        }
    )
    out_path = os.path.join(data_dir, "cell_stats.csv")
    with open(out_path, "w") as outfile:
        outfile.write(stats_df.to_csv(index=False))
    return out_path


def _iterate_covfile(cov_file, c_col, p_col, m_col, u_col, coverage, sep, header):
    try:
        if cov_file.name.lower().endswith(".gz"):
            # handle gzip-compressed file
            lines = gzip.decompress(cov_file.read()).decode().strip().split("\n")
            if header:
                lines = lines[1:]
            for line in lines:
                yield _line_to_values(
                    line.strip().split(sep), c_col, p_col, m_col, u_col, coverage
                )
        else:
            # handle uncompressed file
            if header:
                _ = cov_file.readline()
            for line in cov_file:
                yield _line_to_values(
                    line.decode().strip().split(sep),
                    c_col,
                    p_col,
                    m_col,
                    u_col,
                    coverage,
                )
    # we add the name of the file which caused the crash so that the user can fix it
    except Exception as e:
        raise type(e)(f"{e} (in file: {cov_file.name})").with_traceback(
            sys.exc_info()[2]
        )


def _line_to_values(line, c_col, p_col, m_col, u_col, coverage):
    chrom = line[c_col]
    pos = int(line[p_col])
    n_meth = int(line[m_col])
    if coverage:
        n_unmeth = int(line[u_col]) - n_meth
    else:
        n_unmeth = int(line[u_col])
    return chrom, pos, n_meth, n_unmeth
