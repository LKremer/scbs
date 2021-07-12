import os
import sys
import gzip
import glob
import pandas as pd
import numpy as np
import numba
import scipy.sparse as sp_sparse
import click
from statsmodels.stats.proportion import proportion_confint
from numba import njit, prange
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from umap import UMAP
from scbs.utils import echo, secho


# ignore division by 0 and division by NaN error
np.seterr(divide="ignore", invalid="ignore")


def _line_to_values(line, c_col, p_col, m_col, u_col, coverage):
    chrom = line[c_col]
    pos = int(line[p_col])
    n_meth = int(line[m_col])
    if coverage:
        n_unmeth = int(line[u_col]) - n_meth
    else:
        n_unmeth = int(line[u_col])
    return chrom, pos, n_meth, n_unmeth



def _write_summary_stats(data_dir, cell_names, n_obs, n_meth):
    stats_df = pd.DataFrame(
        {
            "cell_name": cell_names,
            "n_obs": n_obs,
            "n_meth": n_meth,
            "global_meth_frac": np.divide(n_meth, n_obs),
        }
    )
    out_path = os.path.join(data_dir, "cell_stats.csv")
    with open(out_path, "w") as outfile:
        outfile.write(stats_df.to_csv(index=False))
    return out_path



class Smoother(object):
    def __init__(self, sparse_mat, bandwidth=1000, weigh=False):
        # create the tricube kernel
        self.hbw = bandwidth // 2
        rel_dist = np.abs((np.arange(bandwidth) - self.hbw) / self.hbw)
        self.kernel = (1 - (rel_dist ** 3)) ** 3
        # calculate (unsmoothed) methylation fraction across the chromosome
        n_obs = sparse_mat.getnnz(axis=1)
        n_meth = np.ravel(np.sum(sparse_mat > 0, axis=1))
        self.mfracs = np.divide(n_meth, n_obs)
        self.cpg_pos = (~np.isnan(self.mfracs)).nonzero()[0]
        assert n_obs.shape == n_meth.shape == self.mfracs.shape
        if weigh:
            self.weights = np.log1p(n_obs)
        self.weigh = weigh
        return

    def smooth_whole_chrom(self):
        smoothed = {}
        for i in self.cpg_pos:
            window = self.mfracs[i - self.hbw : i + self.hbw]
            nz = ~np.isnan(window)
            try:
                k = self.kernel[nz]
                if self.weigh:
                    w = self.weights[i - self.hbw : i + self.hbw][nz]
                    smooth_val = np.divide(np.sum(window[nz] * k * w), np.sum(k * w))
                else:
                    smooth_val = np.divide(np.sum(window[nz] * k), np.sum(k))
                smoothed[i] = smooth_val
            except IndexError:
                # when the smoothing bandwith is out of bounds of
                # the chromosome... needs fixing eventually
                smoothed[i] = np.nan
        return smoothed


def smooth(data_dir, bandwidth, use_weights):
    out_dir = os.path.join(data_dir, "smoothed")
    os.makedirs(out_dir, exist_ok=True)
    for mat_path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        chrom = os.path.basename(os.path.splitext(mat_path)[0])
        echo(f"Reading chromosome {chrom} data from {mat_path} ...")
        mat = sp_sparse.load_npz(mat_path)
        sm = Smoother(mat, bandwidth, use_weights)
        echo(f"Smoothing chromosome {chrom} ...")
        smoothed_chrom = sm.smooth_whole_chrom()
        with open(os.path.join(out_dir, f"{chrom}.csv"), "w") as smooth_out:
            for pos, smooth_val in smoothed_chrom.items():
                smooth_out.write(f"{pos},{smooth_val}\n")
    secho(f"\nSuccessfully wrote smoothed methylation data to {out_dir}.", fg="green")
    return


def _output_file_handle(raw_path):
    path = raw_path.lower()
    if path.endswith(".gz"):
        handle = gzip.open(raw_path, "wt")
    elif path.endswith(".csv"):
        handle = open(raw_path, "w")
    else:
        handle = open(raw_path + ".csv", "w")
    return handle


def _load_smoothed_chrom(data_dir, chrom):
    smoothed_path = os.path.join(data_dir, "smoothed", f"{chrom}.csv")
    if not os.path.isfile(smoothed_path):
        raise Exception(
            "Could not find smoothed methylation data for "
            f"chromosome {chrom} at {smoothed_path} . "
            "Please run 'scbs smooth' first."
        )
    typed_dict = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64,
    )
    with open(smoothed_path, "r") as smooth_file:
        for line in smooth_file:
            pos, smooth_val = line.strip().split(",")
            typed_dict[int(pos)] = float(smooth_val)
    return typed_dict


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
                raise Exception(f"{regions} is not sorted alphabetically!")
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


@njit
def _find_peaks(smoothed_vars, swindow_centers, var_cutoff, half_bw):
    """" variance peak calling """
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
):
    # shift windows along the chromosome and calculate the variance for each window.
    windows = np.arange(start, end, stepsize)
    smoothed_var = np.empty(windows.shape, dtype=np.float64)
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
        smoothed_var[i] = np.nanvar(mean_shrunk_resid)
    return windows, smoothed_var


@njit(nogil=True)
def _calc_mean_shrunken_residuals(
    data_chrom,
    indices_chrom,
    indptr_chrom,
    start,
    end,
    smoothed_vals,
    n_cells,
    chrom_len,
    shrinkage_factor=1,
):
    shrunken_resid = np.full(n_cells, np.nan)
    if start > chrom_len:
        return shrunken_resid
    end += 1
    if end > chrom_len:
        end = chrom_len
    # slice the methylation values so that we only keep the values in the window
    data = data_chrom[indptr_chrom[start] : indptr_chrom[end]]
    if data.size == 0:
        # return NaN for regions without coverage or regions without CpGs
        return shrunken_resid
    # slice indices
    indices = indices_chrom[indptr_chrom[start] : indptr_chrom[end]]
    # slice index pointer
    indptr = indptr_chrom[start : end + 1] - indptr_chrom[start]
    indptr_diff = np.diff(indptr)

    n_obs = np.zeros(n_cells, dtype=np.int64)
    n_obs_start = np.bincount(indices)
    n_obs[0 : n_obs_start.shape[0]] = n_obs_start

    meth_sums = np.zeros(n_cells, dtype=np.int64)
    smooth_sums = np.zeros(n_cells, dtype=np.float64)
    cpg_idx = 0
    nobs_cpg = indptr_diff[cpg_idx]
    # nobs_cpg: how many of the next values correspond to the same CpG
    # e.g. a value of 3 means that the next 3 values are of the same CpG
    for i in range(data.shape[0]):
        while nobs_cpg == 0:
            cpg_idx += 1
            nobs_cpg = indptr_diff[cpg_idx]
        nobs_cpg -= 1
        cell_idx = indices[i]
        smooth_sums[cell_idx] += smoothed_vals[start + cpg_idx]
        meth_value = data[i]
        if meth_value == -1:
            continue  # skip 0 meth values when summing
        meth_sums[cell_idx] += meth_value

    for i in range(n_cells):
        if n_obs[i] > 0:
            shrunken_resid[i] = (meth_sums[i] - smooth_sums[i]) / (
                n_obs[i] + shrinkage_factor
            )
    return shrunken_resid


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


@njit
def _count_n_cpg(region_indptr):
    """
    Count the total number of CpGs in a region, based on CSR matrix index pointers.
    """
    prev_val = 0
    n_cpg = 0
    for val in region_indptr:
        if val != prev_val:
            n_cpg += 1
            prev_val = val
    return n_cpg


@njit
def _calc_region_stats(
    data_chrom, indices_chrom, indptr_chrom, start, end, n_cells, chrom_len
):
    n_meth = np.zeros(n_cells, dtype=np.int64)
    n_total = np.zeros(n_cells, dtype=np.int64)
    if start > chrom_len:
        n_obs_cpg = 0
    else:
        end += 1
        if end > chrom_len:
            end = chrom_len
        # slice the methylation values so that we only keep the values in the window
        data = data_chrom[indptr_chrom[start] : indptr_chrom[end]]
        if data.size > 0:
            # slice indices
            indices = indices_chrom[indptr_chrom[start] : indptr_chrom[end]]
            # slice index pointer
            indptr = indptr_chrom[start : end + 1] - indptr_chrom[start]
            n_obs_cpg = _count_n_cpg(indptr)  # total number of CpGs in the region
            for i in range(data.shape[0]):
                cell_i = indices[i]
                meth_value = data[i]
                n_total[cell_i] += 1
                if meth_value == -1:
                    continue
                n_meth[cell_i] += meth_value
    return n_meth, n_total, np.divide(n_meth, n_total), n_obs_cpg


def scan(data_dir, output, bandwidth, stepsize, var_threshold, threads=-1):
    if threads != -1:
        numba.set_num_threads(threads)
    n_threads = numba.get_num_threads()
    half_bw = bandwidth // 2
    # sort chroms by filesize. We start with largest chrom to find the var threshold
    chrom_paths = sorted(
        glob.glob(os.path.join(data_dir, "*.npz")),
        key=lambda x: os.path.getsize(x),
        reverse=True,
    )
    # will be discovered on the largest chromosome based on X% cutoff
    var_threshold_value = None
    for mat_path in chrom_paths:
        chrom = os.path.basename(os.path.splitext(mat_path)[0])
        mat = _load_chrom_mat(data_dir, chrom)
        smoothed_cpg_vals = _load_smoothed_chrom(data_dir, chrom)
        # n_obs = mat.getnnz(axis=1)
        # n_meth = np.ravel(np.sum(mat > 0, axis=1))
        # mfracs = np.divide(n_meth, n_obs)
        chrom_len, n_cells = mat.shape
        cpg_pos_chrom = np.nonzero(mat.getnnz(axis=1))[0]

        if n_threads > 1:
            echo(f"Scanning chromosome {chrom} using {n_threads} parallel threads ...")
        else:
            echo(f"Scanning chromosome {chrom} ...")
        # slide windows along the chromosome and calculate the mean
        # shrunken variance of residuals for each window.
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
        )

        if var_threshold_value is None:
            # this is the first=biggest chrom, so let's find our variance threshold here
            var_threshold_value = np.nanquantile(window_variances, 1 - var_threshold)
            echo(f"Determined the variance threshold of {var_threshold_value}.")

        peak_starts, peak_ends = _find_peaks(
            window_variances, genomic_positions, var_threshold_value, half_bw
        )

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


def imputing_pca(
    X, n_components=10, n_iterations=10, scale_features=True, center_features=True
):
    # center and scale features
    X = scale(X, axis=0, with_mean=center_features, with_std=scale_features)
    # for each set of predicted values, we calculated how similar it is to the values
    # we predicted in the previous iteration, so that we can roughly see when our
    # prediction converges
    dist = np.full(n_iterations, fill_value=np.nan)
    # varexpl = np.full(n_iterations, fill_value=np.nan)
    nan_positions = np.isnan(X)
    X[nan_positions] = 0  # zero is our first guess for all missing values
    # start iterative imputation
    for i in range(n_iterations):
        echo(f"PCA iteration {i + 1}...")
        previous_guess = X[nan_positions]  # what we imputed in the previous iteration
        # PCA on the imputed matrix
        pca = PCA(n_components=n_components)
        pca.fit(X)
        # impute missing values with PCA
        new_guess = (pca.inverse_transform(pca.transform(X)))[nan_positions]
        X[nan_positions] = new_guess
        # compare our new imputed values to the ones from the previous round
        dist[i] = np.mean((previous_guess - new_guess) ** 2)
        # varexpl[i] = np.sum(pca.explained_variance_ratio_)
    pca.prediction_dist_iter = dist  # distance between predicted values
    # pca.total_var_exp_iter = varexpl
    pca.X_imputed = X
    return pca


def reduce(
    matrix,  # filepath to a matrix produced by scbs matrix OR pandas DataFrame
    value_column="shrunken_residual",  # the name of the column containing the methylation values
    center_cells=False,  # subtract the mean methylation from each cell? (should be useful for GpC accessibility data)
    max_na_region=0.7,  # maximum allowed fraction of missing values for each region. Set this to 1 to prevent filtering.
    max_na_cell=0.95,  # maximum allowed fraction of missing values for each cell (note that this threshold is applied after regions were filtered)
    n_pc=10,  # number of principal components to compute
    n_iterations=10,  # number of iterations for PCA imputation
    n_neighbors=20,  # a umap parameter
    min_dist=0.1,  # a umap parameter
):
    """
    Takes the output of 'scbs matrix' and reduces it to fewer dimensions, first by
    PCA and then by UMAP.
    """
    if isinstance(matrix, str):
        df = pd.read_csv(matrix, header=0)
    elif isinstance(matrix, pd.core.frame.DataFrame):
        df = matrix
    else:
        raise Exception("'matrix' must be either a path or a pandas DataFrame.")
    # make a proper matrix (cell x region)
    echo("Converting long matrix to wide matrix...")
    df_wide = (
        df.assign(
            region=df.apply(
                lambda x: f"{x['chromosome']}:{x['start']}-{x['end']}", axis=1
            )
        )
        .loc[:, ["cell_name", "region", value_column]]
        .pivot(index="cell_name", columns="region", values=value_column)
    )
    X = np.array(df_wide)
    Xdim_old = X.shape
    # filter regions that were not observed in many cells
    na_frac_region = np.sum(np.isnan(X), axis=0) / X.shape[0]
    X = X[:, na_frac_region <= max_na_region]
    echo(f"filtered {Xdim_old[1] - X.shape[1]} of {Xdim_old[1]} regions.")
    # filter cells that did not observe many regions
    na_frac_cell = np.sum(np.isnan(X), axis=1) / X.shape[1]
    X = X[na_frac_cell <= max_na_cell, :]
    echo(f"filtered {Xdim_old[0] - X.shape[0]} of {Xdim_old[0]} cells.")
    cell_names = df_wide.index[na_frac_cell <= max_na_cell]
    percent_missing = na_frac_cell[na_frac_cell <= max_na_cell] * 100
    # optionally: for each value, subtract the mean methylation of that cell
    # this is mostly for GpC-accessibility data since some cells may receive more
    # methylase than others
    if center_cells:
        X = scale(X, axis=1, with_mean=True, with_std=False)
        echo("centered cells.")
    # run our modified PCA
    echo(f"running modified PCA ({n_pc=}, {n_iterations=})...")
    pca = imputing_pca(X, n_components=n_pc, n_iterations=n_iterations)
    X_pca_reduced = pca.transform(pca.X_imputed)
    echo(f"running UMAP ({n_neighbors=}, {min_dist=})...")
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    X_umap_reduced = reducer.fit_transform(X_pca_reduced)
    # generate output table as pandas df
    col_names = ["UMAP" + str(i + 1) for i in range(X_umap_reduced.shape[1])] + [
        "PC" + str(i + 1) for i in range(X_pca_reduced.shape[1])
    ]
    out_df = pd.DataFrame(
        data=np.concatenate((X_umap_reduced, X_pca_reduced), axis=1),
        index=cell_names,
        columns=col_names,
    )
    out_df["percent_missing"] = percent_missing
    return out_df, pca
