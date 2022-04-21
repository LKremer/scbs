import gzip
import os
from glob import glob

import numba
import numpy as np
from numba import njit, prange
import pandas as pd
import math

from .numerics import _calc_mean_shrunken_residuals
from .smooth import _load_smoothed_chrom
from .utils import _check_data_dir, _load_chrom_mat, echo, secho

# ignore division by 0 and division by NaN error
np.seterr(divide="ignore", invalid="ignore")


def _output_file_handle(raw_path):
    path = raw_path.lower()
    if path.endswith(".gz"):
        handle = gzip.open(raw_path, "wt")
    elif path.endswith(".csv"):
        handle = open(raw_path, "w")
    else:
        handle = open(raw_path + ".csv", "w")
    return handle


#@njit
def _find_peaks(smoothed_vars, swindow_centers, var_cutoff, half_bw):
    """
    After we calculated the variance for each window, this function
    merges overlapping windows above a variance threshold (var_cutoff).
    Returns the start and end coordinates of the bigger merged peaks.
    """
    peak_starts = []
    peak_ends = []
    in_peak = False
    for var, pos in zip(smoothed_vars, swindow_centers):
        if var > var_cutoff:
            if not in_peak:
                # entering new peak
                in_peak = True
                if peak_ends and pos - half_bw <= max(peak_ends):
                    # it's not really a new peak, the last peak wasn't
                    # finished, there was just a small dip...
                    peak_ends.pop()
                else:
                    peak_starts.append(pos - half_bw)
        else:
            if in_peak:
                # exiting peak
                in_peak = False
                peak_ends.append(pos + half_bw)
    if in_peak:
        peak_ends.append(pos)
    assert len(peak_starts) == len(peak_ends)
    return peak_starts, peak_ends

def welch_t_test(group1, group2, len_group1, len_group2):
    s2_group1 = np.nanvar(group1)
    s2_group2 = np.nanvar(group2)
    s_delta = math.sqrt(s2_group1/len_group1 + s2_group2/len_group2)
    t = (np.nanmean(group1) - np.nanmean(group2)) / s_delta
    df = (s2_group1/len_group1 + s2_group2/len_group2)**2 / ((s2_group1/len_group1)**2/(len_group1-1)) + ((s2_group2/len_group2)**2/(len_group2-1))
    return t, s_delta, df

#@njit(parallel=True)
def _move_windows(
    start,
    end,
    stepsize,
    half_bw,
    data_chrom,
    indices_chrom,
    indptr_chrom,
    smoothed_vals,
    n_cells,
    chrom_len,
    group1_index,
    group2_index
):
    """
    Move the sliding window along the whole chromosome.
    For each window, calculate the mean shrunken residuals,
    i.e. 1 methylation value per cell for that window. Then
    calculate the variance of shrunken residuals. This is our
    measure of methylation variability for that window.
    """
    windows = np.arange(start, end, stepsize)
    t = np.empty(windows.shape, dtype=np.float64)
    s_delta = np.empty(windows.shape, dtype=np.float64)
    df = np.empty(windows.shape, dtype=np.float64)
    for i in prange(windows.shape[0]):
        pos = windows[i]
        mean_shrunk_resid = _calc_mean_shrunken_residuals(
            data_chrom,
            indices_chrom,
            indptr_chrom,
            pos - half_bw,
            pos + half_bw,
            smoothed_vals,
            n_cells,
            chrom_len,
        )

        group1 = mean_shrunk_resid[group1_index.flatten()]
        group2 = mean_shrunk_resid[group2_index.flatten()]

        len_group1 = np.count_nonzero(~np.isnan(group1))
        len_group2 = np.count_nonzero(~np.isnan(group2))

        t_test = welch_t_test(group1, group2, len_group1, len_group2)
        print(t_test[0], t_test[1], t_test[2], pos)

        #pos = np.where(windows == [i])
        #t[pos[0]] = t_test[0]
        #s_delta[pos[0]] = t_test[1]
        #df[pos[0]] = t_test[2]

    return ç, t, s_delta, df


def scan(data_dir, output, bandwidth, stepsize, var_threshold, threads=-1):
<<<<<<< Updated upstream
=======
    #exclude the following later on. uses pandas and it needs to be possible to chose groups
    celltypes = pd.read_csv("/Users/marti/Documents/M-Thesis/documents-export-2022-04-06/all_celltypes.csv", sep=',')
    celltypes = celltypes.to_numpy()
    group1_index = celltypes == 'neuroblast'
    group2_index = celltypes == 'oligodendrocyte'
>>>>>>> Stashed changes
    _check_data_dir(data_dir, assert_smoothed=True)
    if threads != -1:
        numba.set_num_threads(threads)
    n_threads = numba.get_num_threads()
    half_bw = bandwidth // 2
    # sort chroms by filesize. We start with largest chrom to find the var threshold
    chrom_paths = sorted(
        glob(os.path.join(data_dir, "*.npz")),
        key=lambda x: os.path.getsize(x),
        reverse=True,
    )

    # the variance threshold will be determined based on the largest chromosome.
    # by default, we take the 98th percentile of all window variances.
    var_threshold_value = None
    for mat_path in chrom_paths:
        chrom = os.path.basename(os.path.splitext(mat_path)[0])
        mat = _load_chrom_mat(data_dir, chrom)
        smoothed_cpg_vals = _load_smoothed_chrom(data_dir, chrom)
        chrom_len, n_cells = mat.shape
        cpg_pos_chrom = np.nonzero(mat.getnnz(axis=1))[0]

        if n_threads > 1:
            echo(f"Scanning chromosome {chrom} using {n_threads} parallel threads ...")
        else:
            echo(f"Scanning chromosome {chrom} ...")
        # slide windows along the chromosome and calculate the variance of
        # mean shrunken residuals for each window.
        start = cpg_pos_chrom[0] + half_bw + 1
        end = cpg_pos_chrom[-1] - half_bw - 1
        genomic_positions, window_variances = _move_windows(
            start,
            end,
            stepsize,
            half_bw,
            mat.data,
            mat.indices,
            mat.indptr,
            smoothed_cpg_vals,
            n_cells,
            chrom_len,
            group1_index,
            group2_index
        )

        if var_threshold_value is None:
            # this is the first=biggest chrom, so let's find our variance threshold here
            var_threshold_value = np.nanquantile(window_variances, 1 - var_threshold)
            echo(f"Determined the variance threshold of {var_threshold_value}.")

        # merge overlapping windows with high variance, to get bigger regions
        # of variable size
        peak_starts, peak_ends = _find_peaks(
            window_variances, genomic_positions, var_threshold_value, half_bw
        )

        # for each big merged peak, re-calculate the variance and
        # write it to a file.
        for ps, pe in zip(peak_starts, peak_ends):
            peak_var = np.nanvar(
                _calc_mean_shrunken_residuals(
                    mat.data,
                    mat.indices,
                    mat.indptr,
                    ps,
                    pe,
                    smoothed_cpg_vals,
                    n_cells,
                    chrom_len,
                )
            )
            bed_entry = f"{chrom}\t{ps}\t{pe}\t{peak_var}\n"
            output.write(bed_entry)
        if len(peak_starts) > 0:
            secho(
                f"Found {len(peak_starts)} variable regions on chromosome {chrom}.",
                fg="green",
            )
        else:
            secho(
                f"Found no variable regions on chromosome {chrom}.",
                fg="red",
            )
    return


# currently not needed but could be useful:
# @njit
# def _count_n_cells(region_indices):
#     """
#     Count the total number of CpGs in a region, based on CSR matrix indices.
#     Only CpGs that have coverage in at least 1 cell are counted.
#     """
#     seen_cells = set()
#     n_cells = 0
#     for cell_idx in region_indices:
#         if cell_idx not in seen_cells:
#             seen_cells.add(cell_idx)
#             n_cells += 1
#     return n_cells
