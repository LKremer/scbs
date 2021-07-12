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
