from __future__ import print_function, division
import os
import gzip
import glob
import pandas as pd
import numpy as np
import scipy.sparse as sp_sparse
from statsmodels.stats.proportion import proportion_confint
from click import echo, secho

# from numba import prange, jit, njit, vectorize


# ignore division by 0 and division by NaN error
np.seterr(divide="ignore", invalid="ignore")


def _get_filepath(f):
    """ returns the path of a file handle, if needed """
    if type(f) is tuple and hasattr(f[0], "name"):
        return f"{f[0].name} and {len(f) - 1} more files"
    return f.name if hasattr(f, "name") else f


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


def _redefine_bed_regions(start, end, extend_by):
    """
    truncates or extend_bys a region to match the desired length
    """
    center = (start + end) // 2  # take center of region
    new_start = center - extend_by  # bounds = center += half region size
    new_end = center + extend_by
    return new_start, new_end


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


def _parse_cell_names(data_dir):
    cell_names = []
    with open(os.path.join(data_dir, "column_header.txt"), "r") as col_heads:
        for line in col_heads:
            cell_names.append(line.strip())
    return cell_names


def profile(data_dir, regions, output, width, strand_column, label):
    """
    see 'scbs profile --help'
    """
    cell_names = _parse_cell_names(data_dir)
    extend_by = width // 2
    n_regions = 0  # count the total number of valid regions in the bed file
    n_empty_regions = 0  # count the number of regions that don't overlap a CpG
    observed_chroms = set()
    unknown_chroms = set()
    prev_chrom = None
    for bed_entries in _iter_bed(regions, strand_col_i=strand_column):
        chrom, start, end, is_rev_strand, _ = bed_entries
        if chrom in unknown_chroms:
            continue
        if chrom != prev_chrom:
            # we reached a new chrom, load the next matrix
            if chrom in observed_chroms:
                raise Exception(f"{_get_filepath(regions)} is not sorted!")
            mat = _load_chrom_mat(data_dir, chrom)
            if mat is None:
                unknown_chroms.add(chrom)
                continue
            echo(f"extracting methylation for regions on chromosome {chrom} ...")
            observed_chroms.add(chrom)
            if prev_chrom is None:
                # this happens at the very start, i.e. on the first chromosome
                n_cells = mat.shape[1]
                # two empty matrices will collect the number of methylated
                # CpGs and the total CpG count for every position of every
                # cell,summed over all regions
                n_meth_global = np.zeros((extend_by * 2, n_cells), dtype=np.uint32)
                n_non_na_global = np.zeros((extend_by * 2, n_cells), dtype=np.uint32)
                if strand_column:
                    n_meth_global_rev = np.zeros(
                        (extend_by * 2, n_cells), dtype=np.uint32
                    )
                    n_non_na_global_rev = np.zeros(
                        (extend_by * 2, n_cells), dtype=np.uint32
                    )
            prev_chrom = chrom

        # adding half width on both sides of the center of the region
        new_start, new_end = _redefine_bed_regions(start, end, extend_by)

        region = mat[new_start:new_end, :]
        if region.shape[0] != extend_by * 2:
            echo(
                f"skipping region {chrom}:{start}-{end} for now... "
                "out of bounds when extended... Not implemented yet!"
            )
            continue
        n_regions += 1
        if region.nnz == 0:
            # skip regions that contain no CpG
            n_empty_regions += 1
            continue

        n_meth_region = (region > 0).astype(np.uint32)
        n_non_na_region = (region != 0).astype(np.uint32)

        if not is_rev_strand:
            # handle forward regions or regions without strand info
            n_meth_global = n_meth_global + n_meth_region
            n_non_na_global = n_non_na_global + n_non_na_region
        else:
            # handle regions on the minus strand
            n_meth_global_rev = n_meth_global_rev + n_meth_region
            n_non_na_global_rev = n_non_na_global_rev + n_non_na_region

    if strand_column:
        echo("adding regions from minus strand")
        assert n_meth_global_rev.max() > 0
        assert n_non_na_global_rev.max() > 0
        n_meth_global = n_meth_global + np.flipud(n_meth_global_rev)
        n_non_na_global = n_non_na_global + np.flipud(n_non_na_global_rev)

    secho(f"\nSuccessfully profiled {n_regions} regions.", fg="green")
    echo(
        f"{n_empty_regions} of these regions "
        f"({np.divide(n_empty_regions, n_regions):.2%}) "
        f"were not observed in any cell."
    )

    if unknown_chroms:
        secho("\nWarning:", fg="red")
        echo(
            "The following chromosomes are present in "
            f"'{_get_filepath(regions)}' but not in "
            f"'{_get_filepath(data_dir)}':"
        )
        for uc in sorted(unknown_chroms):
            echo(uc)

    # write final output file of binned methylation fractions
    _write_profile(output, n_meth_global, n_non_na_global, cell_names, extend_by, label)
    return


def _get_cell_names(cov_files):
    """
    Use the file base names (without extension) as cell names
    """
    names = []
    for file_handle in cov_files:
        f = file_handle.name
        if f.lower().endswith(".gz"):
            # remove .xxx.gz
            names.append(os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0])
        else:
            # remove .xxx
            names.append(os.path.splitext(os.path.basename(f))[0])
    if len(set(names)) < len(names):
        s = (
            "\n".join(names) + "\nThese sample names are not unique, "
            "check your file names again!"
        )
        raise Exception(s)
    return names


def _iterate_covfile(cov_file, c_col, p_col, m_col, u_col, coverage, sep, header):
    if cov_file.name.lower().endswith(".gz"):
        # handle gzip-compressed file
        lines = gzip.decompress(cov_file.read()).decode().strip().split("\n")
        if header:
            lines = lines[1:]
        for line in lines:
            yield _line_to_values(line.strip().split(sep), c_col, p_col, m_col, u_col, coverage)
    else:
        # handle uncompressed file
        if header:
            _ = cov_file.readline()
        for line in cov_file:
            yield _line_to_values(line.decode().strip().split(sep), c_col, p_col, m_col, u_col, coverage)


def _write_column_names(output_dir, cell_names, fname="column_header.txt"):
    """
    The column names (usually cell names) will be
    written to a separate text file
    """
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as col_head:
        for cell_name in cell_names:
            col_head.write(cell_name + "\n")
    return out_path

def _human_to_computer(file_format):
    if len(file_format) == 1:
        if file_format[0].lower() in ('bismarck', 'bismark'):
            c_col, p_col, m_col, u_col, coverage, sep, header = 0, 1, 4, 5, False, '\t', False
        elif file_format[0].lower() == 'allc':
            c_col, p_col, m_col, u_col, coverage, sep, header = 0, 1, 4, 5, True, '\t', True
        else:
            raise Exception("Format not correct. Check --help for further information.", fg="red")
    elif len(file_format) == 6:
        c_col = int(file_format[0])-1
        p_col = int(file_format[1])-1
        m_col = int(file_format[2])-1
        u_col = int(file_format[3][0:-1])-1
        info = file_format[3][-1].lower()
        if info =='c':
            coverage = True
        elif info =='m':
            coverage = False
        else: 
            raise Exception("Format for column with coverage/methylation must be an integer and either c for coverage or m for methylation (eg 4c)", fg="red") 
        sep = str(file_format[4])
        if sep == '\\t':
            sep = '\t'
        header = bool(int(file_format[5]))
    else: 
        raise Exception("Format not correct. Check --help for further information.", fg="red")
    return c_col, p_col, m_col, u_col, coverage, sep, header

def _line_to_values(line, c_col, p_col, m_col, u_col, coverage):
    chrom = line[c_col]
    pos = int(line[p_col])
    n_meth = int(line[m_col])
    if coverage:
        n_unmeth = int(line[u_col])-n_meth
    else:
        n_unmeth = int(line[u_col])
    return chrom, pos, n_meth, n_unmeth

def _dump_coo_files(fpaths, input_format, n_cells, header, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    #c_col, p_col, m_col, u_col = [f - 1 for f in input_format]
    c_col, p_col, m_col, u_col, coverage, sep, header = _human_to_computer(input_format.split(':'))
    coo_files = {}
    chrom_sizes = {}
    for cell_n, cov_file in enumerate(fpaths):
        if cell_n % 50 == 0:
            echo("{0:.2f}% done...".format(100 * cell_n / n_cells))
        for line_vals in _iterate_covfile(cov_file, c_col, p_col, m_col, u_col, coverage, sep, header):
            chrom, genomic_pos, n_meth, n_unmeth = line_vals
            if n_meth != 0 and n_unmeth != 0:
                continue  # currently we ignore all CpGs that are not "clear"!
            meth_value = 1 if n_meth > 0 else -1
            if chrom not in coo_files:
                coo_path = os.path.join(output_dir, f"{chrom}.coo")
                coo_files[chrom] = open(coo_path, "w")
                chrom_sizes[chrom] = 0
            if genomic_pos > chrom_sizes[chrom]:
                chrom_sizes[chrom] = genomic_pos
            coo_files[chrom].write(f"{genomic_pos},{cell_n},{meth_value}\n")
    for fhandle in coo_files.values():
        # maybe somehow use try/finally or "with" to make sure
        # they're closed even when crashing
        fhandle.close()
    echo("100% done.")
    return coo_files, chrom_sizes


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


def prepare(input_files, data_dir, input_format, header):
    cell_names = _get_cell_names(input_files)
    n_cells = len(cell_names)
    # we use this opportunity to count some basic summary stats
    n_obs_cell = np.zeros(n_cells, dtype=np.int64)
    n_meth_cell = np.zeros(n_cells, dtype=np.int64)

    # For each chromosome, we first make a sparse matrix in COO (coordinate)
    # format, because COO can be constructed value by value, without knowing the
    # dimensions beforehand. This means we can construct it cell by cell.
    # We dump the COO to hard disk to save RAM and then later convert each COO to a
    # more efficient format (CSR).
    echo(f"Processing {n_cells} methylation files...")
    coo_files, chrom_sizes = _dump_coo_files(
        input_files, input_format, n_cells, header, data_dir
    )
    echo(
        "\nStoring methylation data in 'compressed "
        "sparse row' (CSR) matrix format for future use."
    )

    # read each COO file and convert the matrix to CSR format.
    for chrom in coo_files.keys():
        # create empty matrix
        chrom_size = chrom_sizes[chrom]
        echo(f"Populating {chrom_size} x {n_cells} matrix for chromosome {chrom}...")
        # populate with values from temporary COO file
        coo_path = os.path.join(data_dir, f"{chrom}.coo")
        mat_path = os.path.join(data_dir, f"{chrom}.npz")
        coo = np.loadtxt(coo_path, delimiter=",")
        mat = sp_sparse.coo_matrix(
            (coo[:, 2], (coo[:, 0], coo[:, 1])),
            shape=(chrom_size + 1, n_cells),
            dtype=np.int8,
        )
        echo(f"Converting from COO to CSR...")
        mat = mat.tocsr()  # convert from COO to CSR format

        n_obs_cell += mat.getnnz(axis=0)
        n_meth_cell += np.ravel(np.sum(mat > 0, axis=0))

        echo(f"Writing to {mat_path} ...")
        sp_sparse.save_npz(mat_path, mat)
        os.remove(coo_path)  # delete temporary .coo file

    colname_path = _write_column_names(data_dir, cell_names)
    echo(f"\nWrote matrix column names to {colname_path}")
    stats_path = _write_summary_stats(data_dir, cell_names, n_obs_cell, n_meth_cell)
    echo(f"Wrote summary stats for each cell to {stats_path}")
    secho(
        f"\nSuccessfully stored methylation data for {n_cells} cells "
        f"with {len(coo_files.keys())} chromosomes.",
        fg="green",
    )
    return


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
    out_dict = {}
    with open(smoothed_path, "r") as smooth_file:
        for line in smooth_file:
            pos, smooth_val = line.strip().split(",")
            out_dict[int(pos)] = float(smooth_val)
    return out_dict


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
        "cell_name",
        "n_meth",
        "n_obs," "meth_frac",
        "residual",
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
            else:
                echo(f"extracting methylation for regions on chromosome {chrom} ...")
                smoothed_cpg_vals = _load_smoothed_chrom(data_dir, chrom)
            observed_chroms.add(chrom)
            prev_chrom = chrom

        region = mat[start:end, :]
        n_regions += 1
        if region.nnz == 0:
            # skip regions that contain no CpG
            n_empty_regions += 1
            continue

        n_obs = region.getnnz(axis=0)
        n_meth = np.ravel(np.sum(region > 0, axis=0))
        nz_cells = np.nonzero(n_obs > 0)[0]
        # smoothing and centering:
        cpg_positions = np.nonzero(region.getnnz(axis=1))[0]
        region = region[cpg_positions, :].todense()  # remove all non-CpG bases
        region = np.where(region == 0, np.nan, region)
        region = np.where(region == -1, 0, region)
        # load smoothed values
        smoothed = np.array([smoothed_cpg_vals[start + c] for c in cpg_positions])
        # centering, using the smoothed means
        mvals_resid = np.subtract(region, np.reshape(smoothed, (-1, 1)))
        # calculate methylation % per cell (3 different ways, to compare performance)
        mfracs = np.nanmean(region, axis=0)
        resid = np.nanmean(mvals_resid, axis=0)
        resid_shrunk = np.nansum(mvals_resid, axis=0) / (n_obs + 1)

        # write "count" table
        for c in nz_cells:
            out_vals = [
                chrom,
                start,
                end,
                cell_names[c],
                n_meth[c],
                n_obs[c],
                mfracs[c],
                resid[c],
                resid_shrunk[c],
            ]
            if keep_other_columns and other_columns:
                out_vals += other_columns
            output.write(",".join(str(v) for v in out_vals) + "\n")

    echo(f"Profiled {n_regions} regions.\n")
    if n_empty_regions / n_regions > 0.5:
        secho("Warning - most regions have no coverage in any cell:", fg="red")
    echo(
        f"{n_empty_regions} regions ({n_empty_regions/n_regions:.2%}) "
        f"contained no covered methylation site."
    )
    return


def _calc_residual_var(mat, mfracs, smoothed_vals, start, end):
    """
    Calculate the variance of shrunken residuals for a given windows
    """
    window = mfracs[start:end]
    nz = ~np.isnan(window)
    if not np.sum(nz):
        return np.nan

    region = mat[start:end, :]
    n_obs = region.getnnz(axis=0)
    region = region[nz, :].todense()  # remove all non-CpG bases
    region = np.where(region == 0, np.nan, region)
    region = np.where(region == -1, 0, region)

    cpg_pos = np.arange(start, end)
    smooth_avg = np.array([smoothed_vals[p] for p in cpg_pos[nz]])

    resid = np.subtract(region, np.reshape(smooth_avg, (-1, 1)))
    assert np.nansum(resid, axis=0).shape == n_obs.shape
    resid_shrunk = np.nansum(resid, axis=0) / (n_obs + 1)
    return np.nanvar(resid_shrunk)


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


def scan(data_dir, output, bandwidth, stepsize, var_threshold, chromosome):
    half_bw = bandwidth // 2
    if chromosome:
        # run on a single chromosome
        chrom_paths = [os.path.join(data_dir, f"{chromosome}.npz")]
        secho(f"Searching only on chromosome {chromosome}", fg="green")
    else:
        # run on all chromosomes
        chrom_paths = sorted(
            glob.glob(os.path.join(data_dir, "*.npz")),
            key=lambda x: os.path.getsize(x),
            reverse=True,
        )

    var_threshold_value = (
        None  # will be discovered on the largest chromosome based on X% cutoff
    )
    for mat_path in chrom_paths:
        if chromosome:
            chrom = chromosome
        else:
            chrom = os.path.basename(os.path.splitext(mat_path)[0])
        mat = _load_chrom_mat(data_dir, chrom)
        smoothed_cpg_vals = _load_smoothed_chrom(data_dir, chrom)
        n_obs = mat.getnnz(axis=1)
        n_meth = np.ravel(np.sum(mat > 0, axis=1))
        mfracs = np.divide(n_meth, n_obs)
        cpg_pos_chrom = np.nonzero(mat.getnnz(axis=1))[0]

        # shift windows along the chromosome and calculate the variance for each window.
        # this is very slow but could be very fast with numba! easy to parallelize
        start = cpg_pos_chrom[0] + half_bw + 1
        end = cpg_pos_chrom[-1] - half_bw - 1
        smoothed_var = []
        genomic_pos = []
        for pos in range(start, end, stepsize):
            genomic_pos.append(pos)
            sm = _calc_residual_var(
                mat, mfracs, smoothed_cpg_vals, pos - half_bw, pos + half_bw
            )
            smoothed_var.append(sm)
            if len(smoothed_var) % 100_000 == 0:
                echo(
                    f"chromosome {chrom} is {100 * (pos-start) / (end-start):.3}% "
                    "scanned."
                )

        if var_threshold_value is None:
            # this is the first=biggest chromosome, so let's find our variance threshold here
            var_threshold_value = np.nanquantile(smoothed_var, 1 - var_threshold)
            echo(f"Determined the variance threshold of {var_threshold_value}.")

        peak_starts, peak_ends = _find_peaks(
            smoothed_var, genomic_pos, var_threshold_value, half_bw
        )

        for ps, pe in zip(peak_starts, peak_ends):
            peak_var = _calc_residual_var(mat, mfracs, smoothed_cpg_vals, ps, pe)
            bed_entry = f"{chrom}\t{ps}\t{pe}\t{peak_var}\n"
            output.write(bed_entry)
        if len(peak_starts) > 0:
            secho(
                f"Wrote {len(peak_starts)} variable bins to {output.name}.", fg="green"
            )
        else:
            secho(
                f"Found no variable regions on chromosome {chrom}.",
                fg="red",
            )
    return
