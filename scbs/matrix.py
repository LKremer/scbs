from .utils import _iter_bed, echo, secho, _parse_cell_names, _load_chrom_mat
from .smooth import _load_smoothed_chrom
from .numerics import _calc_region_stats, _calc_mean_shrunken_residuals
import numpy as np


def matrix(
    data_dir,
    regions,
    output,
    keep_other_columns=False,
):
    output_header = [
        "chromosome",
        "start",
        "end",
        "n_sites",
        "n_cells",
        "cell_name",
        "n_meth",
        "n_obs",
        "meth_frac",
        "shrunken_residual",
    ]
    cell_names = _parse_cell_names(data_dir)
    n_regions = 0  # count the total number of valid regions in the bed file
    n_empty_regions = 0  # count the number of regions that don't overlap a CpG
    observed_chroms = set()
    unknown_chroms = set()
    prev_chrom = None

    for bed_entries in _iter_bed(regions, keep_cols=keep_other_columns):
        chrom, start, end, _, other_columns = bed_entries
        if prev_chrom is None:
            # only happens once on the very first bed entry: write header
            if other_columns and keep_other_columns:
                output_header += [f"bed_col{i + 4}" for i in range(len(other_columns))]
            output.write(",".join(output_header) + "\n")
        if chrom in unknown_chroms:
            continue
        if chrom != prev_chrom:
            # we reached a new chrom, load the next matrix
            if chrom in observed_chroms:
                raise Exception(
                    f"{regions} is not sorted alphabetically! "
                    "Please use 'bedtools sort'"
                )
            mat = _load_chrom_mat(data_dir, chrom)
            if mat is None:
                unknown_chroms.add(chrom)
                observed_chroms.add(chrom)
                prev_chrom = chrom
                continue  # skip this region
            else:
                echo(f"extracting methylation for regions on chromosome {chrom} ...")
                smoothed_vals = _load_smoothed_chrom(data_dir, chrom)
                chrom_len, n_cells = mat.shape
                observed_chroms.add(chrom)
                prev_chrom = chrom
        # calculate methylation fraction, shrunken residuals etc. for the region:
        n_regions += 1
        n_meth, n_total, mfracs, n_obs_cpgs = _calc_region_stats(
            mat.data, mat.indices, mat.indptr, start, end, n_cells, chrom_len
        )
        nz_cells = np.nonzero(n_total > 0)[0]  # index of cells that observed the region
        n_obs_cells = nz_cells.shape[0]  # in how many cells we observed the region
        if nz_cells.size == 0:
            # skip regions that were not observed in any cell
            n_empty_regions += 1
            continue
        resid_shrunk = _calc_mean_shrunken_residuals(
            mat.data,
            mat.indices,
            mat.indptr,
            start,
            end,
            smoothed_vals,
            n_cells,
            chrom_len,
        )
        # write "count" table
        for c in nz_cells:
            out_vals = [
                chrom,
                start,
                end,
                n_obs_cpgs,
                n_obs_cells,
                cell_names[c],
                n_meth[c],
                n_total[c],
                mfracs[c],
                resid_shrunk[c],
            ]
            if keep_other_columns and other_columns:
                out_vals += other_columns
            output.write(",".join(str(v) for v in out_vals) + "\n")

    if n_regions == 0:
        raise Exception("bed file contains no regions.")
    echo(f"Profiled {n_regions} regions.\n")
    if (n_empty_regions / n_regions) > 0.5:
        secho("Warning - most regions have no coverage in any cell:", fg="red")
    echo(
        f"{n_empty_regions} regions ({n_empty_regions/n_regions:.2%}) "
        f"contained no covered methylation site."
    )
    return
