from __future__ import print_function, division
import os
import gzip
import glob
import pandas as pd
import numpy as np
import scipy.sparse as sp_sparse
import json
from statsmodels.stats.proportion import proportion_confint
from click import echo, secho, style


# ignore division by 0 and division by NaN error
np.seterr(divide="ignore", invalid="ignore")


def _get_filepath(f):
    """ returns the path of a file handle, if needed """
    if type(f) is tuple and hasattr(f[0], "name"):
        return f"{f[0].name} and {len(f) - 1} more files"
    return f.name if hasattr(f, "name") else f


def _iter_bed(file_obj, strand_col_i=None):
    is_rev_strand = False
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
        # yield chrom, start, end, and whether the feature is on the minus strand
        yield values[0], int(values[1]), int(values[2]), is_rev_strand


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


def profile(data_dir, regions, output, width, strand_column, label):
    """
    see 'scbs profile --help'
    """
    cell_names = []
    with open(os.path.join(data_dir, "column_header.txt"), "r") as col_heads:
        for line in col_heads:
            cell_names.append(line.strip())

    extend_by = width // 2
    n_regions = 0  # count the total number of valid regions in the bed file
    n_empty_regions = 0  # count the number of regions that don't overlap a CpG
    observed_chroms = set()
    unknown_chroms = set()
    prev_chrom = None
    for bed_entries in _iter_bed(regions, strand_col_i=strand_column):
        chrom, start, end, is_rev_strand = bed_entries
        if chrom in unknown_chroms:
            continue
        if chrom != prev_chrom:
            # we reached a new chrom, load the next matrix
            if chrom in observed_chroms:
                raise Exception(f"{_get_filepath(regions)} is not sorted!")
            mat_path = os.path.join(data_dir, f"{chrom}.npz")
            try:
                echo(f"loading chromosome {chrom} from {mat_path} ... ")
                mat = sp_sparse.load_npz(mat_path)
                # echo("done!")
            except FileNotFoundError:
                secho(f"The file {mat_path} does not exist", fg="red")
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


def _iterate_covfile(cov_file, header):
    """
    Iterate over a single cell input file, line by line,
    optionally skipping the header and unzipping on the fly
    """
    if cov_file.name.lower().endswith(".gz"):
        # handle gzip-compressed file
        lines = gzip.decompress(cov_file.read()).decode().strip().split("\n")
        if header:
            lines = lines[1:]
        for line in lines:
            values = line.strip().split("\t")
            yield values
    else:
        # handle uncompressed file
        if header:
            _ = cov_file.readline()
        for line in cov_file:
            values = line.decode().strip().split("\t")
            yield values


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


def _dump_dok_files(fpaths, input_format, n_cells, header, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    c_col, p_col, m_col, u_col = [f - 1 for f in input_format]
    dok_files = {}
    chrom_sizes = {}
    for cell_n, cov_file in enumerate(fpaths):
        if cell_n % 100 == 0:
            echo("{0:.2f}% done...".format(100 * cell_n / n_cells))
        for line_vals in _iterate_covfile(cov_file, header):
            n_meth, n_unmeth = int(line_vals[m_col]), int(line_vals[u_col])
            if n_meth != 0 and n_unmeth != 0:
                continue  # currently we ignore all CpGs that are not "clear"!
            meth_value = 1 if n_meth > 0 else -1
            genomic_pos = int(line_vals[p_col])
            chrom = line_vals[c_col]
            if chrom not in dok_files:
                dok_path = os.path.join(output_dir, f"{chrom}.dok")
                dok_files[chrom] = open(dok_path, "w")
                chrom_sizes[chrom] = 0
            if genomic_pos > chrom_sizes[chrom]:
                chrom_sizes[chrom] = genomic_pos
            dok_files[chrom].write(f"{genomic_pos},{cell_n},{meth_value}\n")
    for fhandle in dok_files.values():
        # maybe somehow use try/finally or "with" to make sure
        # they're closed even when crashing
        fhandle.close()
    echo("100% done.")
    return dok_files, chrom_sizes


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

    # For each chromosome, we first make a sparse matrix in DOK (dictionary of keys)
    # format, because DOK can be constructed value by value, without knowing the
    # dimensions beforehand. This means we can construct it cell by cell.
    # We dump the DOK to hard disk to save RAM and then later convert each DOK to a
    # more efficient format (CSR).
    echo(f"Processing {n_cells} methylation files...")
    dok_files, chrom_sizes = _dump_dok_files(
        input_files, input_format, n_cells, header, data_dir
    )
    echo(
        "\nStoring methylation data in 'compressed "
        "sparse row' (CSR) matrix format for future use."
    )

    # read each DOK file and convert the matrix to CSR format.
    for chrom in dok_files.keys():
        # create empty matrix
        chrom_size = chrom_sizes[chrom]
        mat = sp_sparse.dok_matrix((chrom_size + 1, n_cells), dtype=np.int8)
        echo(f"Populating {chrom_size} x {n_cells} matrix for chromosome {chrom}...")
        # populate with values from temporary dok file
        dok_path = os.path.join(data_dir, f"{chrom}.dok")
        with open(dok_path, "r") as fhandle:
            for line in fhandle:
                genomic_pos, cell_n, meth_value = line.strip().split(",")
                mat[int(genomic_pos), int(cell_n)] = int(meth_value)
        mat_path = os.path.join(data_dir, f"{chrom}.npz")
        echo(f"Writing to {mat_path} ...")
        mat = mat.tocsr()  # convert from DOK to CSR format

        n_obs_cell += mat.getnnz(axis=0)
        n_meth_cell += np.ravel(np.sum(mat > 0, axis=0))

        sp_sparse.save_npz(mat_path, mat)
        os.remove(dok_path)  # delete temporary .dok file

    colname_path = _write_column_names(data_dir, cell_names)
    echo(f"\nWrote matrix column names to {colname_path}")
    stats_path = _write_summary_stats(data_dir, cell_names, n_obs_cell, n_meth_cell)
    echo(f"Wrote summary stats for each cell to {stats_path}")
    secho(
        f"\nSuccessfully stored methylation data for {n_cells} cells "
        f"with {len(dok_files.keys())} chromosomes.",
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
        smoothed = np.full(self.cpg_pos.shape, fill_value=np.nan)
        for j, i in enumerate(self.cpg_pos):
            window = self.mfracs[i - self.hbw : i + self.hbw]
            nz = ~np.isnan(window)
            try:
                k = self.kernel[nz]
                if self.weigh:
                    w = self.weights[i - self.hbw : i + self.hbw][nz]
                    smooth_val = np.divide(np.sum(window[nz] * k * w), np.sum(k * w))
                else:
                    smooth_val = np.divide(np.sum(window[nz] * k), np.sum(k))
                smoothed[j] = smooth_val
            except IndexError:
                # when the smoothing bandwith is out of bounds of
                # the chromosome... needs fixing eventually
                smoothed[j] = np.nan
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
        np.save(os.path.join(out_dir, f"{chrom}.npy"), smoothed_chrom)
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


def matrix(data_dir, regions, output, keep_cols=False, bandwidth=2000, use_weights=False):
    cell_names = []
    with open(os.path.join(data_dir, "column_header.txt"), "r") as col_heads:
        for line in col_heads:
            cell_names.append(line.strip())

    n_regions = 0  # count the total number of valid regions in the bed file
    n_empty_regions = 0  # count the number of regions that don't overlap a CpG
    observed_chroms = set()
    unknown_chroms = set()
    prev_chrom = None

    # with _output_file_handle(output) as out:
    for bed_entries in _iter_bed(regions):
        chrom, start, end = bed_entries[:3]
        if chrom in unknown_chroms:
            continue
        # further_bed_entries = ",".join(bed_entries[3:])
        if chrom != prev_chrom:
            # we reached a new chrom, load the next matrix
            if chrom in observed_chroms:
                raise Exception(f"{regions} is not sorted alphabetically!")
            mat_path = os.path.join(data_dir, f"{chrom}.npz")
            smoothed_path = os.path.join(data_dir, "smoothed", f"{chrom}.npy")
            try:
                print(f"loading chromosome {chrom} from {mat_path} ...", end="\r")
                mat = sp_sparse.load_npz(mat_path)
                print(f"loading chromosome {chrom} from {mat_path} ... done!")
                print(
                    f"extracting methylation for regions on chromosome {chrom} ..."
                )
                smoothed_cpg_pos = np.load(smoothed_path)
            except FileNotFoundError:
                unknown_chroms.add(chrom)
                print(
                    f"  WARNING: Couldn't load {mat_path}! "
                    f"Regions on chromosome {chrom} will not be used."
                )
            observed_chroms.add(chrom)
            prev_chrom = chrom

        region = mat[start:end, :]
        n_regions += 1
        if region.nnz == 0:
            # skip regions that contain no CpG
            n_empty_regions += 1
            continue

        n_non_na = region.getnnz(axis=0)
        n_meth = np.ravel(np.sum(region > 0, axis=0))
        nz_cells = np.nonzero(n_non_na > 0)[0]
        # smoothing and centering:
        cpg_positions = np.nonzero(region.getnnz(axis=1))[0]
        region = region[cpg_positions, :].todense()  # remove all non-CpG bases
        region = np.where(region==0, np.nan, region)
        region = np.where(region==-1, 0, region)
        # smoothing:
        smooth_start = np.searchsorted(cpg_pos_chrom, start, "left")
        smooth_end = np.searchsorted(cpg_pos_chrom, end, "right")
        smoothed = smoothed_cpg_pos[smooth_start:smooth_end]
        # centering, using the smoothed means
        mvals_centered = np.subtract(region, np.reshape(smoothed, (-1, 1)))
        # add a 0 pseudocount as a sort of shrinkage
        mvals_pcount = np.zeros((mvals_centered.shape[0]+1, mvals_centered.shape[1]))
        mvals_pcount[1:, :] = mvals_centered
        # calculate methylation % per cell (3 different ways, so we can compare performance)
        mfracs_raw = np.nanmean(region, axis=0)
        mfracs_centered = np.nanmean(mvals_centered, axis=0)
        mfracs_pcount = np.nanmean(mvals_pcount, axis=0)

        for c in nz_cells:
            if keep_cols:
                s = f"{chrom},{start},{end},{cell_names[c]},{n_meth[c]},{n_non_na[c]},{mfracs_raw[c]},{mfracs_centered[c]},{mfracs_pcount[c]},{further_bed_entries}\n"
            else:
                # s = f"{chrom}:{start}-{end},{cell_i},{cell_names[c]},{n_meth_cell},{n_non_na_cell},{mfrac}\n"
                s = f"{chrom},{start},{end},{cell_names[c]},{n_meth[c]},{n_non_na[c]},{mfracs_raw[c]},{mfracs_centered[c]},{mfracs_pcount[c]}\n"
            output.write(s)

    print(
        f"Profiled {n_regions} regions.\n"
        f"{n_empty_regions} regions ({n_empty_regions/n_regions:.2%}) "
        f"contained no covered methylation site."
    )
    return
