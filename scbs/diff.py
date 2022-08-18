import math
import os
from glob import glob

import numba
import numpy as np
from numba import njit, prange

from .numerics import _calc_mean_shrunken_residuals
from .smooth import _load_smoothed_chrom
from .utils import _check_data_dir, _load_chrom_mat, echo
from .scbs import _find_peaks

# ignore division by 0 and division by NaN error
np.seterr(divide="ignore", invalid="ignore")

np.random.seed(5)


def permuted_indices(idx_celltypes, celltype_1, celltype_2, total_cells):
    permutation = np.random.permutation(idx_celltypes)
    while (permutation == idx_celltypes).all():
        permutation = np.random.permutation(idx_celltypes)

    index_g1 = np.zeros(total_cells, dtype=bool)
    index_g2 = np.zeros(total_cells, dtype=bool)

    for i in permutation[celltype_1]:
        index_g1[i] = True
    for i in permutation[celltype_2]:
        index_g2[i] = True

    return index_g1, index_g2


@njit
def calc_fdr(datatype):
    fdisc = 0
    tdisc = 0
    adj_p_val_arr = np.empty(datatype.shape, dtype=np.float64)

    for i in range(len(datatype)):
        if datatype[i]:
            tdisc += 1

        else:
            fdisc += 1

        adj_p_val = fdisc / (fdisc + tdisc)
        adj_p_val_arr[i] = adj_p_val

    return adj_p_val_arr


@njit(nogil=True)
def welch_t_test(group1, group2, min_cells):
    len_g1 = len(group1)
    if len_g1 < min_cells:
        return np.nan, np.nan, np.nan
    len_g2 = len(group2)
    if len_g2 < min_cells:
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
    denominator = ((var_g1 / len_g1) ** 2 / (len_g1 - 1)) + (
        (var_g2 / len_g2) ** 2 / (len_g2 - 1)
    )
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
    index,
    min_cells,
    datatype,
):
    """
    Move the sliding window along the whole chromosome.
    For each window, calculate the mean shrunken residuals,
    i.e. 1 methylation value per cell for that window. Then
    calculate the t-statistic of shrunken residuals. This is our
    measure of differential methylation for that window.
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

        # the real indices are stored in the first "column".
        if datatype == "real":
            group1 = mean_shrunk_resid[index[0, 0]]
            group1 = group1[~np.isnan(group1)]

            group2 = mean_shrunk_resid[index[1, 0]]
            group2 = group2[~np.isnan(group2)]

        # A new permutation is used every 2Mbp.
        # Therefore, the next "column" of the array has to be accessed.
        # The column index (chrom_bin)
        # equals the window position on the chromosome separated by 2Mbp.
        else:
            chrom_bin = pos // 2000000 + 1
            group1 = mean_shrunk_resid[index[0, chrom_bin]]
            group1 = group1[~np.isnan(group1)]

            group2 = mean_shrunk_resid[index[1, chrom_bin]]
            group2 = group2[~np.isnan(group2)]

        t, s_delta, df = welch_t_test(group1, group2, min_cells)
        t_array[i] = t

    return windows, t_array


def calc_tstat_peaks(
    chrom,
    datatype,
    cells,
    half_bw,
    data_chrom,
    indices_chrom,
    indptr_chrom,
    smoothed_vals,
    n_cells,
    chrom_len,
    index,
    min_cells,
    threshold_datatype,
    window_tstat_groups,
    genomic_positions,
):
    output = []
    for _ in range(6):
        array = np.empty(0)
        output.append(array)

    for tstat_windows, cell, threshold_value in zip(
        window_tstat_groups, cells, threshold_datatype
    ):
        # merge overlapping windows with lowest and highest t-statistic,
        # to get bigger regions of variable size
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
            # the real indices are stored in the first "column".
            if datatype == "real":
                group1 = mean_shrunk_resid[index[0, 0]]
                group1 = group1[~np.isnan(group1)]

                group2 = mean_shrunk_resid[index[1, 0]]
                group2 = group2[~np.isnan(group2)]

            # A new permutation is used every 2Mbp.
            # Therefore, the next "column" of the array has to be accessed.
            # The column index (chrom_bin)
            # equals the middle of the merged peak separated by 2Mbp.
            else:
                chrom_bin = int(((pe - ps) / 2 + ps) // 2000000 + 1)
                group1 = mean_shrunk_resid[index[0, chrom_bin]]
                group1 = group1[~np.isnan(group1)]

                group2 = mean_shrunk_resid[index[1, chrom_bin]]
                group2 = group2[~np.isnan(group2)]

            t_stat, s_delta, df = welch_t_test(group1, group2, min_cells)

            datapoints = [chrom, ps, pe, t_stat, datatype, cell]

            for datapoint in range(len(datapoints)):
                output[datapoint] = np.append(output[datapoint], datapoints[datapoint])

    return output


def diff(
    data_dir,
    cell_file,
    output,
    bandwidth,
    stepsize,
    threshold,
    min_cells,
    threads=-1,
    debug=False,
):
    celltypes = np.loadtxt(cell_file, dtype=str)
    cells = []

    for i in range(len(celltypes)):
        if str(celltypes[i]) != "nan":
            cells.append(celltypes[i])

    # which cell types are present in the input file (defines the two groups)
    cells = np.unique(cells)

    index_realg1 = (celltypes == cells[0]).flatten()
    index_realg2 = (celltypes == cells[1]).flatten()

    # needs to be calculated for the permutation
    idx_celltypes = np.asarray(
        np.where((celltypes == cells[0]) | (celltypes == cells[1]))
    ).flatten()
    celltype_1 = (
        celltypes[(celltypes == cells[0]) | (celltypes == cells[1])] == cells[0]
    )
    celltype_2 = (
        celltypes[(celltypes == cells[0]) | (celltypes == cells[1])] == cells[1]
    )
    total_cells = len(celltypes)

    _check_data_dir(data_dir, assert_smoothed=True)
    if threads != -1:
        numba.set_num_threads(threads)
    n_threads = numba.get_num_threads()
    half_bw = bandwidth // 2

    # sort chroms by file size. We start with largest chrom to find the threshold
    chrom_paths = sorted(
        glob(os.path.join(data_dir, "*.npz")),
        key=lambda x: os.path.getsize(x),
        reverse=True,
    )

    threshold_values = np.zeros([2, 2])
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

        # Permute data every 2Mbp.
        # Store necessary number of permutations for
        # the according chromosome length in an array.
        # While scanning the chromosome every 2 Mbp another permutation
        # ("column" of the array) can be used.
        n_perm = chrom_len // 2000000 + 1
        perm_idx = np.empty([2, n_perm + 1, len(celltypes)], dtype=bool)
        # first "column" holds the real perm_idx of both cell types
        perm_idx[0][0] = index_realg1
        perm_idx[1][0] = index_realg2
        for i in range(n_perm):
            perm_idx[0][i + 1], perm_idx[1][i + 1] = permuted_indices(
                idx_celltypes, celltype_1, celltype_2, total_cells
            )

        output_chrom = []
        datatypes = ["real", "permuted"]
        for datatype in datatypes:
            start = cpg_pos_chrom[0] + half_bw + 1
            end = cpg_pos_chrom[-1] - half_bw - 1
            genomic_positions, window_tstat = _move_windows(
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
                perm_idx,
                min_cells,
                datatype,
            )

            # calculate t-statistic for group1 (low t-values)
            # and group 2 (high t-values)
            # to make this possible the t-stat was multiplied by -1 for group1
            window_tstat_groups = [window_tstat * -1, window_tstat]

            # This is the first=biggest chrom,
            # so let's find our t-statistic threshold here.
            # The t-statistic threshold will be determined
            # based on the largest chromosome.
            # By default, we take the 98th percentile of all window t-statistics.
            # Thresholds for real indices are stored in the first
            # and for permuted indices in the second array.
            iteration = 0
            if threshold_values[1, 1] == 0:
                for tstat_windows, cell in zip(window_tstat_groups, cells):
                    if datatype == "real":
                        threshold_values[0, iteration] = np.nanquantile(
                            tstat_windows, 1 - threshold
                        )
                        echo(
                            f"Determined threshold of {threshold_values[0,iteration]} "
                            f"for {cell} of {datatype} data."
                        )
                    if datatype == "permuted":
                        threshold_values[1, iteration] = np.nanquantile(
                            tstat_windows, 1 - threshold
                        )

                        if debug:
                            echo(
                                f"Determined threshold of "
                                f"{threshold_values[1,iteration]} "
                                f"for {cell} of {datatype} data."
                            )
                    iteration += 1

            if datatype == "real":
                threshold_datatype = threshold_values[0]

            if datatype == "permuted":
                threshold_datatype = threshold_values[1]

            # calculate t-statistic for merged peaks
            output_iteration = calc_tstat_peaks(
                chrom,
                datatype,
                cells,
                half_bw,
                mat.data,
                mat.indices,
                mat.indptr,
                smoothed_cpg_vals,
                n_cells,
                chrom_len,
                perm_idx,
                min_cells,
                threshold_datatype,
                window_tstat_groups,
                genomic_positions,
            )

            # join outputs of all loops
            if len(output_chrom) == 0:
                output_chrom = output_iteration
            else:
                for i in range(len(output_chrom)):
                    output_chrom[i] = np.append(output_chrom[i], output_iteration[i])

        # join outputs of individual chromosomes
        if len(output_final) == 0:
            output_final = output_chrom
        else:
            for column in range(len(output_final)):
                output_final[column] = np.append(
                    output_final[column], output_chrom[column]
                )

    # sort descending by absolute t-statistic
    idx = np.argsort(np.absolute(output_final[3]) * -1)
    for column in range(len(output_final)):
        output_final[column] = output_final[column][idx]

    # calculate FDR / adjusted p-values
    adj_p_val = calc_fdr(output_final[4] == "real")
    output_final.append(adj_p_val)

    output_final[1] = output_final[1].astype("int")
    output_final[2] = output_final[2].astype("int")

    if not debug:
        # remove permuted values and datatype array
        filter_datatype = output_final[4] == "real"
        for column in range(len(output_final)):
            output_final[column] = output_final[column][filter_datatype]
        del output_final[4]

        differential = np.count_nonzero(output_final[5] < 0.05)
        echo(f"found {differential} significant differentially methylated regions.")

    np.savetxt(output, np.transpose(output_final), delimiter="\t", fmt="%s")

    """
    # only important if there is two output files
    # generate list of arrays according to selected celltypes
    # and remove array with celltypes
    # save both outputs
    # if datatype array is removed again change index from 5 to 4
    outputs = [output1, output2]
    for cell, output in zip(cells, outputs):
        output_cell = []
        filter_cells = output_final[5] == cell
        for column in range(len(output_final)):
            output_cell.append(output_final[column][filter_cells])
        del output_cell[5]

        np.savetxt(output, np.transpose(output_cell), delimiter = "\t", fmt = '%s')
    """

    return
