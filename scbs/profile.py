import numpy as np
import scipy.sparse as sp_sparse
import os
import sys
import gzip


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


def _parse_cell_names(data_dir):
    cell_names = []
    with open(os.path.join(data_dir, "column_header.txt"), "r") as col_heads:
        for line in col_heads:
            cell_names.append(line.strip())
    return cell_names


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


def _redefine_bed_regions(start, end, extend_by):
    """
    truncates or extend_bys a region to match the desired length
    """
    center = (start + end) // 2  # take center of region
    new_start = center - extend_by  # bounds = center += half region size
    new_end = center + extend_by
    return new_start, new_end


def _iterate_covfile(cov_file, c_col, p_col, m_col, u_col, coverage, sep, header):
    try:
        if cov_file.name.lower().endswith(".gz"):
            # handle gzip-compressed file
            lines = gzip.decompress(cov_file.read()).decode().strip().split("\n")
            if header:
                lines = lines[1:]
            for line in lines:
                yield _line_to_values(
                    line.strip().split(sep), c_col, p_col, m_col, u_col, coverage
                )
        else:
            # handle uncompressed file
            if header:
                _ = cov_file.readline()
            for line in cov_file:
                yield _line_to_values(
                    line.decode().strip().split(sep),
                    c_col,
                    p_col,
                    m_col,
                    u_col,
                    coverage,
                )
    # we add the name of the file which caused the crash so that the user can fix it
    except Exception as e:
        raise type(e)(f"{e} (in file: {cov_file.name})").with_traceback(
            sys.exc_info()[2]
        )
