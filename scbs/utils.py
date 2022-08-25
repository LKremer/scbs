import os
from glob import glob

import click
import scipy.sparse as sp_sparse


# print messages go to stderr
# output file goes to stdout (when using "-" as output file)
# that way you can pipe the output file e.g. into bedtools
def echo(*args, **kwargs):
    click.echo(*args, **kwargs, err=True)
    return


def secho(*args, **kwargs):
    click.secho(*args, **kwargs, err=True)
    return


def _get_filepath(f):
    """returns the path of a file handle, if needed"""
    if type(f) is tuple and hasattr(f[0], "name"):
        return f"{f[0].name} and {len(f) - 1} more files"
    return f.name if hasattr(f, "name") else f


def _iter_bed(file_obj, strand_col_i=None, keep_cols=False):
    is_rev_strand = False
    other_columns = False
    if strand_col_i is not None:
        strand_col_i -= 1  # CLI is 1-indexed
    for line in file_obj:
        if line.startswith("#") or not line.strip():
            continue  # skip comments
        values = line.strip().split("\t")
        if strand_col_i is not None:
            strand_val = values[strand_col_i]
            if strand_val == "-" or strand_val == "-1":
                is_rev_strand = True
            elif strand_val == "+" or strand_val == "1":
                is_rev_strand = False
            else:
                raise Exception(
                    f"Invalid strand column value '{strand_val}'. "
                    "Should be '+', '-', '1', or '-1'."
                )
        if keep_cols:
            other_columns = values[3:]
        # yield chrom, start, end, and whether the feature is on the minus strand
        yield values[0], int(values[1]), int(values[2]), is_rev_strand, other_columns


def _parse_cell_names(data_dir):
    cell_names = []
    with open(os.path.join(data_dir, "column_header.txt"), "r") as col_heads:
        for line in col_heads:
            cell_name = line.strip()
            if cell_name:
                cell_names.append(cell_name)
    return cell_names


def _load_chrom_mat(data_dir, chrom):
    mat_path = os.path.join(data_dir, f"{chrom}.npz")
    echo(f"loading chromosome {chrom} from {mat_path} ...")
    try:
        mat = sp_sparse.load_npz(mat_path)
    except FileNotFoundError:
        secho("Warning: ", fg="red", nl=False)
        echo(
            f"Couldn't load methylation data for chromosome {chrom} from {mat_path}. "
            f"Regions on chromosome {chrom} will not be considered."
        )
        mat = None
    return mat


def _check_data_dir(data_dir, assert_smoothed=False):
    """
    quickly peek into data_dir to make sure the user
    did not specify an empty directory
    """
    npz_files = glob(os.path.join(data_dir, "*.npz"))
    if not npz_files:
        raise Exception(
            f"Your specified DATA_DIR '{data_dir}' is invalid since it does not "
            "contain any chromosome files.\n           Chromosome files "
            "end in '.npz' and are automatically created by 'scbs prepare'."
        )
    if assert_smoothed:
        smooth_files = glob(os.path.join(data_dir, "smoothed", "*.csv"))
        if not smooth_files:
            raise Exception(
                f"Your specified DATA_DIR '{data_dir}' is not smoothed yet."
                "\n           Please smooth your data with 'scbs smooth'."
            )
