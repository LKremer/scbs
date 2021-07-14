import click
import pandas as pd
import numpy as np
import gzip
import sys
import os
import scipy.sparse as sp_sparse
from statsmodels.stats.proportion import proportion_confint


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
    """ returns the path of a file handle, if needed """
    if type(f) is tuple and hasattr(f[0], "name"):
        return f"{f[0].name} and {len(f) - 1} more files"
    return f.name if hasattr(f, "name") else f


def _write_profile(
    output_file, n_meth_global, n_non_na_global, cell_names, extend_by, add_column
):
    """
    write the whole profile to a csv table in long format
    """
    output_path = _get_filepath(output_file)
    echo("Converting to long table format...")
    n_total_vals = (
        pd.DataFrame(n_non_na_global)
        .reset_index()
        .melt("index", var_name="cell", value_name="n_total")
        .get("n_total")
    )

    long_df = (
        pd.DataFrame(n_meth_global)
        .reset_index()
        .melt("index", var_name="cell", value_name="n_meth")
        .assign(cell_name=lambda x: [cell_names[c] for c in x["cell"]])
        .assign(index=lambda x: np.subtract(x["index"], extend_by))
        .assign(cell=lambda x: np.add(x["cell"], 1))
        .assign(n_total=n_total_vals)
        .loc[lambda df: df["n_total"] > 0, :]
        .assign(meth_frac=lambda x: np.divide(x["n_meth"], x["n_total"]))
        .rename(columns={"index": "position"})
    )

    echo("Calculating Agresti-Coull confidence interval...")
    ci = proportion_confint(
        long_df["n_meth"], long_df["n_total"], method="agresti_coull"
    )

    echo(f"Writing output to {output_path}...")
    long_df = long_df.assign(ci_lower=ci[0]).assign(ci_upper=ci[1])

    if add_column:
        long_df = long_df.assign(label=add_column)

    output_file.write(long_df.to_csv(index=False))
    return


def _iter_bed(file_obj, strand_col_i=None, keep_cols=False):
    is_rev_strand = False
    other_columns = False
    if strand_col_i is not None:
        strand_col_i -= 1  # CLI is 1-indexed
    for line in file_obj:
        if line.startswith("#"):
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
            cell_names.append(line.strip())
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
