import numpy as np
import gzip
import os
import scipy.sparse as sp_sparse
import sys
import pandas as pd
from .utils import echo, secho


def prepare(input_files, data_dir, input_format):
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
    coo_files, chrom_sizes = _dump_coo_files(
        input_files, input_format, n_cells, data_dir
    )
    echo(
        "\nStoring methylation data in 'compressed "
        "sparse row' (CSR) matrix format for future use."
    )

    # read each COO file and convert the matrix to CSR format.
    for chrom in coo_files.keys():
        # create empty matrix
        chrom_size = chrom_sizes[chrom]
        echo(f"Populating {chrom_size} x {n_cells} matrix for chromosome {chrom}...")
        # populate with values from temporary COO file
        coo_path = os.path.join(data_dir, f"{chrom}.coo")
        mat_path = os.path.join(data_dir, f"{chrom}.npz")
        mat = _load_csr_from_coo(coo_path, chrom_size, n_cells)
        n_obs_cell += mat.getnnz(axis=0)
        n_meth_cell += np.ravel(np.sum(mat > 0, axis=0))

        echo(f"Writing to {mat_path} ...")
        sp_sparse.save_npz(mat_path, mat)
        os.remove(coo_path)  # delete temporary .coo file

    colname_path = _write_column_names(data_dir, cell_names)
    echo(f"\nWrote matrix column names to {colname_path}")
    stats_path = _write_summary_stats(data_dir, cell_names, n_obs_cell, n_meth_cell)
    echo(f"Wrote summary stats for each cell to {stats_path}")
    secho(
        f"\nSuccessfully stored methylation data for {n_cells} cells "
        f"with {len(coo_files.keys())} chromosomes.",
        fg="green",
    )
    return


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
                coo_files[chrom] = open(coo_path, "w")
                chrom_sizes[chrom] = 0
            if genomic_pos > chrom_sizes[chrom]:
                chrom_sizes[chrom] = genomic_pos
            coo_files[chrom].write(f"{genomic_pos},{cell_n},{meth_value}\n")
    for fhandle in coo_files.values():
        # maybe somehow use try/finally or "with" to make sure
        # they're closed even when crashing
        fhandle.close()
    echo("100% done.")
    return coo_files, chrom_sizes


def _load_csr_from_coo(coo_path, chrom_size, n_cells):
    try:
        coo = pd.read_csv(coo_path, delimiter=",", header=None).values
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


class CoverageFormat:
    """Describes the columns in the coverage file."""

    def __init__(self, chrom, pos, meth, umeth, coverage, sep, header):
        self.chrom = chrom
        self.pos = pos
        self.meth = meth
        self.umeth = umeth
        self.cov = coverage
        self.sep = sep
        self.header = header

    def totuple(self):
        """Transform to use it in non-refactored code for now."""
        return (
            self.chrom,
            self.pos,
            self.meth,
            self.umeth,
            self.cov,
            self.sep,
            self.header,
        )


def create_custom_format(format_string):
    """Create from user specified string."""
    format_string = format_string.lower().split(":")
    if len(format_string) != 6:
        raise Exception("Invalid number of ':'-separated values in custom input format")
    chrom = int(format_string[0]) - 1
    pos = int(format_string[1]) - 1
    meth = int(format_string[2]) - 1
    info = format_string[3][-1]
    if info == "c":
        coverage = True
    elif info == "m":
        coverage = False
    else:
        raise Exception(
            "The 4th column of a custom input format must contain an integer and "
            "either 'c' for coverage or 'm' for methylation (e.g. '4c'), but you "
            f"provided '{format_string[3]}'."
        )
    umeth = int(format_string[3][0:-1]) - 1
    sep = str(format_string[4])
    if sep == "\\t":
        sep = "\t"
    header = bool(int(format_string[5]))
    return CoverageFormat(chrom, pos, meth, umeth, coverage, sep, header)


def create_standard_format(format_name):
    """Create a format object on the basis of the format name."""
    if format_name in ("bismarck", "bismark"):
        return CoverageFormat(
            0,
            1,
            4,
            5,
            False,
            "\t",
            False,
        )
    elif format_name in ("allc", "methylpy"):
        return CoverageFormat(
            0,
            1,
            4,
            5,
            True,
            "\t",
            True,
        )
    else:
        raise Exception(f"{format_name} is not a known format")


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
