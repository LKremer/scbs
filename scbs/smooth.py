import os
import scipy.sparse as sp_sparse
import glob
import numpy as np
from .utils import echo, secho
import numba


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
