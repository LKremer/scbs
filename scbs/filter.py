import os
import glob
import scipy.sparse as sp_sparse
from csv import DictReader
from .utils import echo, secho, _load_chrom_mat, _get_filepath


def _filter_by_name(file, cell_stats_path, keep=True):
    f = _get_filepath(file)
    cells_to_keep_idx = []
    cells_to_keep = {row.strip() for row in file if row.strip()}
    available_cells = set()
    with open(cell_stats_path, "r") as stats_csv:
        reader = DictReader(stats_csv)
        for cell_i, row in enumerate(reader):
            cell_name = row["cell_name"]
            available_cells.add(cell_name)
            if keep:
                if cell_name in cells_to_keep:
                    cells_to_keep_idx.append(cell_i)
            else:  # do not keep those cells
                if cell_name not in cells_to_keep:
                    cells_to_keep_idx.append(cell_i)

    for cell in cells_to_keep:
        if cell not in available_cells:
            raise Exception(
                f"{f} lists cell '{cell}', but it does "
                f"not exist in '{cell_stats_path}'."
            )
    return cells_to_keep_idx


def _filter_by_thresholds(min_sites, max_sites, min_meth, max_meth, cell_stats_path):
    cells_to_keep_idx = []
    counter = {"min-sites": 0, "max-sites": 0, "min-meth": 0, "max-meth": 0}
    with open(cell_stats_path, "r") as stats_csv:
        reader = DictReader(stats_csv)
        for cell_i, row in enumerate(reader):
            n_sites = int(row["n_obs"])
            meth_frac = float(row["global_meth_frac"])
            if min_sites and n_sites < min_sites:
                counter["min-sites"] += 1
                continue
            if max_sites and n_sites > max_sites:
                counter["max-sites"] += 1
                continue
            if min_meth and meth_frac < (min_meth / 100):
                counter["min-meth"] += 1
                continue
            if max_meth and meth_frac > (max_meth / 100):
                counter["max-meth"] += 1
                continue
            cells_to_keep_idx.append(cell_i)
    for threshold, count in counter.items():
        if count:
            echo(f"{count} cells did not pass the --{threshold} threshold.")
    n_cells = cell_i + 1
    n_filtered = n_cells - len(cells_to_keep_idx)
    secho(
        f"\nFiltering {n_filtered} of {n_cells} cells "
        f"({n_filtered/n_cells:.2%})...\n",
        fg="green",
    )
    return cells_to_keep_idx


def _filter_text_file(fpath, rows_to_keep, fpath_out, header=False):
    with open(fpath, "r") as infile, open(fpath_out, "w") as outfile:
        if header:
            header_str = infile.readline()
            outfile.write(header_str)
        for cell_i, row in enumerate(infile):
            if cell_i in rows_to_keep:
                outfile.write(row)


def filter_(
    data_dir, filtered_dir, min_sites, max_sites, min_meth, max_meth, cell_names, keep
):
    stats_path = os.path.join(data_dir, "cell_stats.csv")
    stats_path_out = os.path.join(filtered_dir, "cell_stats.csv")
    colname_path = os.path.join(data_dir, "column_header.txt")
    colname_path_out = os.path.join(filtered_dir, "column_header.txt")
    if cell_names:
        if any([min_sites, max_sites, min_meth, max_meth]):
            secho(
                "Warning: All filtering thresholds (e.g. --min-sites) "
                "will be ignored since you provided --cell-names.\n",
                fg="red",
            )
        cell_idx = _filter_by_name(cell_names, stats_path, keep=keep)
    else:
        if (min_meth and min_meth <= 1) or (max_meth and max_meth <= 1):
            echo(
                "Warning: Your methylation thresholds are very low, "
                "please make sure that you specified a percentage between "
                "0 and 100.",
                fg="red",
            )
        cell_idx = _filter_by_thresholds(
            min_sites, max_sites, min_meth, max_meth, stats_path
        )

    os.makedirs(filtered_dir, exist_ok=True)
    chrom_paths = glob.glob(os.path.join(data_dir, "*.npz"))
    for mat_path in sorted(chrom_paths):
        chrom = os.path.basename(os.path.splitext(mat_path)[0])
        mat = _load_chrom_mat(data_dir, chrom)
        echo(f"Filtering chromosome {chrom}...")
        mat = mat[:, cell_idx]
        mat_path_filtered = os.path.join(filtered_dir, f"{chrom}.npz")
        echo(f"Writing filtered data to {mat_path_filtered} ...")
        sp_sparse.save_npz(mat_path_filtered, mat)

    _filter_text_file(stats_path, cell_idx, stats_path_out, header=True)
    _filter_text_file(colname_path, cell_idx, colname_path_out, header=False)
