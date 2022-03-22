import os
import glob
import scipy.sparse as sp_sparse
from csv import DictReader
from .utils import echo, secho, _load_chrom_mat


def _filter_by_name(keep, cell_stats_path):
    cells_to_keep_idx = []
    cells_to_keep = {row.strip() for row in keep if row.strip()}
    available_cells = set()
    with open(cell_stats_path, "r") as stats_csv:
        reader = DictReader(stats_csv)
        for cell_i, row in enumerate(reader):
            cell_name = row["cell_name"]
            available_cells.add(cell_name)
            if cell_name in cells_to_keep:
                cells_to_keep_idx.append(cell_i)
    for cell in cells_to_keep:
        if cell not in available_cells:
            raise Exception(
                f"You want to keep cell '{cell}', but it does "
                f"not exist in '{cell_stats_path}'."
            )
    return cells_to_keep_idx


def _filter_by_thresholds(min_sites, max_sites, min_meth, max_meth, cell_stats_path):
    cells_to_keep_idx = []
    with open(cell_stats_path, "r") as stats_csv:
        reader = DictReader(stats_csv)
        for cell_i, row in enumerate(reader):
            n_sites = int(row["n_obs"])
            meth_frac = float(row["global_meth_frac"])
            if min_sites and n_sites < min_sites:
                continue
            if max_sites and n_sites > max_sites:
                continue
            if min_meth and meth_frac < (min_meth / 100):
                continue
            if max_meth and meth_frac > (max_meth / 100):
                continue
            cells_to_keep_idx.append(cell_i)
    n_cells = cell_i + 1
    n_filtered = n_cells - len(cells_to_keep_idx)
    secho(
        f"Filtering {n_filtered} of {n_cells} cells " f"({n_filtered/n_cells:.2%})..."
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


def filter_(data_dir, filtered_dir, min_sites, max_sites, min_meth, max_meth, keep):
    stats_path = os.path.join(data_dir, "cell_stats.csv")
    stats_path_out = os.path.join(filtered_dir, "cell_stats.csv")
    colname_path = os.path.join(data_dir, "column_header.txt")
    colname_path_out = os.path.join(filtered_dir, "column_header.txt")
    if keep:
        if any([min_sites, max_sites, min_meth, max_meth]):
            secho(
                "Warning: All filtering thresholds (e.g. --min-sites) "
                "will be ignored since you used --keep.\n",
                fg="red",
            )
        cell_idx = _filter_by_name(keep, stats_path)
    else:
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