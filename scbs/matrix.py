import numba
import os
import numpy as np

from numba import njit, prange

from .numerics import _calc_mean_shrunken_residuals, _calc_region_stats
from .smooth import _load_smoothed_chrom
from .utils import (
    _check_data_dir,
    _iter_bed,
    _load_chrom_mat,
    _parse_cell_names,
    echo,
    secho,
)


@njit(parallel=True)
def calc_mean_mfracs(
    data_chrom,
    indices_chrom,
    indptr_chrom,
    starts,
    ends,
    chrom_len,
    n_cells,
    smoothed_vals,
):
    n_regions = starts.shape[0]
    ends += 1  # include right boundary

    # outputs
    n_meth = np.zeros((n_cells, n_regions), dtype=np.int32)
    n_total = np.zeros((n_cells, n_regions), dtype=np.int32)
    smooth_sums = np.full((n_cells, n_regions), np.nan, dtype=np.float32)

    # 500 regions per thread. it's probably smarter to have one thread per chromosome
    chunk_size = 500
    chunks = np.arange(0, n_regions, chunk_size)
    for chunk_i in prange(chunks.shape[0]):
        chunk_start = chunks[chunk_i]
        chunk_end = chunk_start + chunk_size
        if chunk_end > n_regions:
            chunk_end = n_regions

        for region_i in range(chunk_start, chunk_end):
            start = starts[region_i]
            if start > chrom_len:
                continue  # region is outside of chrom -> ignore
            end = ends[region_i]
            if end > chrom_len:
                end = chrom_len  # region is partially outside of chrom -> truncate
            # slice the methylation values so that we only keep the values in the window
            data = data_chrom[indptr_chrom[start] : indptr_chrom[end]]
            if data.size > 0:
                # slice indices
                indices = indices_chrom[indptr_chrom[start] : indptr_chrom[end]]
                # slice index pointer
                indptr = indptr_chrom[start : end + 1] - indptr_chrom[start]
                indptr_diff = np.diff(indptr)
                cpg_idx = 0
                nobs_cpg = indptr_diff[cpg_idx]
                # nobs_cpg: how many of the next values correspond to the same CpG
                # e.g. a value of 3 means that the next 3 values are of the same CpG
                for i in range(data.shape[0]):
                    while nobs_cpg == 0:
                        cpg_idx += 1
                        nobs_cpg = indptr_diff[cpg_idx]
                    nobs_cpg -= 1
                    cell_i = indices[i]
                    meth_value = data[i]
                    n_total[cell_i, region_i] += 1
                    if np.isnan(smooth_sums[cell_i, region_i]):
                        smooth_sums[cell_i, region_i] = 0.0
                    if meth_value == -1:
                        smooth_sums[cell_i, region_i] -= smoothed_vals[start + cpg_idx]
                        continue
                    else:
                        smooth_sums[cell_i, region_i] += (
                            1 - smoothed_vals[start + cpg_idx]
                        )
                    n_meth[cell_i, region_i] += meth_value
    mean_shrunk_res = smooth_sums / (n_total + 1)
    return n_meth, n_total, mean_shrunk_res


def matrix(
    data_dir,
    regions,
    output,
    threads
):
    os.makedirs(output, exist_ok=True)
    _check_data_dir(data_dir, assert_smoothed=True)
    if threads != -1:
        numba.set_num_threads(threads)
    n_threads = numba.get_num_threads()
    cell_names = _parse_cell_names(data_dir)
    n_regions = 0  # count the total number of valid regions in the bed file
    n_empty_regions = 0  # count the number of regions that don't overlap a CpG
    observed_chroms = set()
    unknown_chroms = set()
    prev_chrom = None

    """
    output matrix of each chromosome will be collected here and merged later
    actually this is a little inefficient, we could just populate one big matrix
    instead of merging the chromosome matrices.
    """
    meth = []
    total = []
    msr = []  # mean shrunken residuals

    for bed_entries in _iter_bed(regions):
        chrom, start, end, *others = bed_entries
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
                if prev_chrom is not None:
                    meth_chrom, total_chrom, msr_chrom = calc_mean_mfracs(
                        mat.data,
                        mat.indices,
                        mat.indptr,
                        np.asarray(starts, dtype=np.int32),
                        np.asarray(ends, dtype=np.int32),
                        chrom_len,
                        n_cells,
                        smoothed_vals,
                    )
                    meth.append(meth_chrom)
                    total.append(total_chrom)
                    msr.append(msr_chrom)
                observed_chroms.add(chrom)
                starts, ends = [], []
                prev_chrom = chrom
        starts.append(start)
        ends.append(end)

    echo(f"Writing matrices to {output} ...")
    meth = np.hstack(meth)
    total = np.hstack(total)
    msr = np.hstack(msr)
    np.savetxt(os.path.join(output, "methylated_sites.csv.gz"), meth, delimiter=",", fmt="%d")
    np.savetxt(os.path.join(output, "total_sites.csv.gz"), total, delimiter=",", fmt="%d")
    np.savetxt(os.path.join(output, "methylation_fractions.csv.gz"), np.divide(meth, total), delimiter=",", fmt="%-.5f")
    np.savetxt(os.path.join(output, "mean_shrunken_residuals.csv.gz"), msr, delimiter=",", fmt="%-.5f")
