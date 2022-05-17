
import gzip
import os
from glob import glob

import numba
import numpy as np
from numba import njit, prange
import pandas as pd
import math
import random

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

def permuted_indices(number, celltypes, cells):
    indices_g1 = []
    indices_g2 = []
    for permutation in range(number):
        permutation = np.random.permutation(celltypes)
        for cell in range(len(cells)):
            index = (permutation == cells[cell]).flatten()
            if cell == 0:
                indices_g1.append(index)
            else:
                indices_g2.append(index)
    return indices_g1, indices_g2

@njit
def calc_fdr_jit(datatype):
    fdisc = 0
    tdisc = 0
    count = 0
    adj_p_val_arr = np.empty(datatype.shape, dtype=np.float64)

    for i in range(len(datatype)):
        if datatype[i]:
            tdisc += 1

        else:
            fdisc += 1

        adj_p_val = fdisc / (fdisc + tdisc)
        adj_p_val_arr[i] = adj_p_val

    return adj_p_val_arr

@njit
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

@njit(nogil=True)
def welch_t_test(group1, group2, length):
    len_g1 = len(group1)
    if len_g1 < length:
        return np.nan, np.nan, np.nan
    len_g2 = len(group2)
    if len_g2 < length:
        return np.nan, np.nan, np.nan

    mean_g1 = np.mean(group1)
    mean_g2 = np.mean(group2)

    sum1 = 0.0
    sum2 = 0.0

    for value in group1:
        sqdif1 = (value - mean_g1) ** 2
        sum1 += sqdif1

    for value in group2:
        sqdif2 = (value - mean_g2) ** 2
        sum2 += sqdif2

    if sum1 == 0.0 and sum2 == 0.0:
        return np.nan, np.nan, np.nan

    var_g1 = sum1 / (len_g1 - 1)
    var_g2 = sum2 / (len_g2 - 1)

    s_delta = math.sqrt(var_g1 / len_g1 + var_g2 / len_g2)
    t = (mean_g1 - mean_g2) / s_delta

    numerator = (var_g1 / len_g1 + var_g2 / len_g2) ** 2
    denominator = ((var_g1 / len_g1) ** 2 / (len_g1 - 1)) + ((var_g2 / len_g2) ** 2 / (len_g2 - 1))
    df = numerator / denominator

    return t, s_delta, df

@njit(parallel=True)
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
    group2_index,
    length
):
    """
    Move the sliding window along the whole chromosome.
    For each window, calculate the mean shrunken residuals,
    i.e. 1 methylation value per cell for that window. Then
    calculate the variance of shrunken residuals. This is our
    measure of methylation variability for that window.
    """
    windows = np.arange(start, end, stepsize)
    t_array = np.empty(windows.shape, dtype=np.float64)
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

        group1 = mean_shrunk_resid[group1_index]
        group1 = group1[~np.isnan(group1)]
        
        group2 = mean_shrunk_resid[group2_index]
        group2 = group2[~np.isnan(group2)]

        t, s_delta, df = welch_t_test(group1, group2, length)

        t_array[i] = t

    return windows, t_array

def calc_tstat_peaks(
        threshold,
        chrom,
        datatype,
        cells,
        cpg_pos_chrom,
        stepsize,
        half_bw,
        data_chrom,
        indices_chrom,
        indptr_chrom,
        smoothed_vals,
        n_cells,
        chrom_len,
        group1_index,
        group2_index,
        length,
        threshold_values
):
    output = []
    for i in range(6):
        array = np.empty(0)
        output.append(array)

    start = cpg_pos_chrom[0] + half_bw + 1
    end = cpg_pos_chrom[-1] - half_bw - 1
    genomic_positions, window_tstat = _move_windows(
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
        group2_index,
        length
    )

    window_tstat_groups = [window_tstat * -1, window_tstat]
    # calculate t-statistic for group1 (low t-values) and group 2 (high t-values)
    # to make this possible the t-stat was multiplied by -1 for group1
    for tstat_windows, cell in zip(window_tstat_groups, cells):
        if len(threshold_values) < 2:
        # this is the first=biggest chrom, so let's find our t-statistic threshold here
            threshold_value = np.nanquantile(tstat_windows, 1 - threshold)
            echo(f"Determined the variance threshold of {threshold_value} for {cell}.")
            threshold_values.append(threshold_value)

    for tstat_windows, cell, threshold_value in zip(window_tstat_groups, cells, threshold_values):
        # merge overlapping windows with lowest and highest t-statistic, to get bigger regions
        # of variable size
        peak_starts, peak_ends = _find_peaks(
            tstat_windows, genomic_positions, threshold_value, half_bw
        )

        # for each big merged peak, re-calculate the t-statistic
        for ps, pe in zip(peak_starts, peak_ends):
            mean_shrunk_resid = _calc_mean_shrunken_residuals(
                data_chrom,
                indices_chrom,
                indptr_chrom,
                ps,
                pe,
                smoothed_vals,
                n_cells,
                chrom_len,
            )
            group1 = mean_shrunk_resid[group1_index]
            group1 = group1[~np.isnan(group1)]

            group2 = mean_shrunk_resid[group2_index]
            group2 = group2[~np.isnan(group2)]

            t_stat, s_delta, df = welch_t_test(group1, group2, length)

            datapoints = [chrom, ps, pe, t_stat, datatype, cell]

            for datapoint in range(len(datapoints)):
                output[datapoint] = np.append(output[datapoint], datapoints[datapoint])

    return output

def diff(data_dir, output1, output2, bandwidth, stepsize, threshold, length = 3, threads=-1):
    #exclude the following later on. uses pandas and it needs to be possible to chose groups
    celltypes = pd.read_csv(os.path.join(data_dir, "celltypes.txt"), header=None).to_numpy()
    cells = ['neuroblast', 'oligodendrocyte']
    # maybe for loop to calc indices and permutations
    group1_index = (celltypes == cells[0]).flatten()
    group2_index = (celltypes == cells[1]).flatten()

    # get indices for both groups from five permutations
    number = 5 #number of permutations, change later. either fixed or argument in diff
    perm_indices_g1, perm_indices_g2 = permuted_indices(number, celltypes, cells) #change number of permutations

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

    # the t-statistic threshold will be determined based on the largest chromosome.
    # by default, we take the 98th percentile of all window variances.
    threshold_values = []
    output_final = []

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

        # slide windows along the chromosome and calculate the t-statistic of
        # mean shrunken residuals for each window.
        output_real = calc_tstat_peaks(
            threshold,
            chrom,
            "real",
            cells,
            cpg_pos_chrom,
            stepsize,
            half_bw,
            mat.data,
            mat.indices,
            mat.indptr,
            smoothed_cpg_vals,
            n_cells,
            chrom_len,
            group1_index,
            group2_index,
            length,
            threshold_values
            )
        # do the same for the permutations
        # generate a list including values for all perumations
        output_perm = []
        for index_g1, index_g2 in zip(perm_indices_g1, perm_indices_g2):
            output = calc_tstat_peaks(
                threshold,
                chrom,
                "permuted",
                cells,
                cpg_pos_chrom,
                stepsize,
                half_bw,
                mat.data,
                mat.indices,
                mat.indptr,
                smoothed_cpg_vals,
                n_cells,
                chrom_len,
                index_g1,
                index_g2,
                length,
                threshold_values
            )
            if len(output_perm) == 0:
                output_perm = output
            else:
                for i in range(len(output_perm)):
                    output_perm[i] = np.append(output_perm[i], output[i])

        # join real and permuted outputs for one chromosome
        output_chrom = []
        for i in range(len(output_real)):
            output_chrom.append(np.append(output_real[i], output_perm[i]))

        # join ouput of individual chromosomes
        if len(output_final) == 0:
            output_final = output_chrom
        else:
            for i in range(len(output_perm)):
                output_final[i] = np.append(output_final[i], output_chrom[i])
                
    # sort descending for absolute t-values
    idx = np.argsort(np.absolute(output_final[3])*-1)
    for coloumn in range(len(output_final)):
        output_final[coloumn] = output_final[coloumn][idx]

    # calculate FDR / adjusted p-values
    adj_p_val = calc_fdr_jit(output_final[4] == "real")
    output_final.append(adj_p_val)

    # remove permuted values and datatype array
    filter_datatype = output_final[4] == "real"
    for coloumn in range(len(output_final)):
        output_final[coloumn] = output_final[coloumn][filter_datatype]
    del output_final[4]

    # generate list of arrays according to selected celltypes and remove array with celltypes
    # save both outputs
    outputs = [output1, output2]
    for cell, output in zip(cells, outputs):
        output_cell = []
        filter_cells = output_final[4] == cell
        for coloumn in range(len(output_final)):
            output_cell.append(output_final[coloumn][filter_cells])
        del output_cell[4]

        differential = np.count_nonzero(output_cell[4] < 0.05)
        echo(f"found {differential} differentially methylated regions for {cell}.")

        np.savetxt(output, np.transpose(output_cell), delimiter = "\t", fmt = '%s')

    return
