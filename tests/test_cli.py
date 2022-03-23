import pytest
import os
import shutil
import scipy.sparse as sp_sparse
from pandas import read_csv
from click.testing import CliRunner
from scbs.cli import cli


def test_prepare_cli(tmp_path):
    runner = CliRunner()
    p = os.path.join(tmp_path, "data_dir")
    result = runner.invoke(cli, ["prepare", "tests/data/tiny/a.cov", "tests/data/tiny/b.cov.gz", p])
    assert result.exit_code == 0, result.output
    mat = sp_sparse.load_npz(os.path.join(p, "1.npz"))
    assert mat.shape == (53, 2)
    assert mat.data.shape == (5,)
    mat = sp_sparse.load_npz(os.path.join(p, "2.npz"))
    assert mat.shape == (1236, 2)
    assert mat.data.shape == (5,)
    with open(os.path.join(p, "cell_stats.csv")) as stats:
        assert stats.read() == ("cell_name,n_obs,n_meth,global_meth_frac\n"
                                "a,5,2,0.4\n"
                                "b,5,3,0.6\n")


def test_smooth_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["smooth", "--bandwidth", "2", "tests/data/tiny/data_dir/"])
    with open("tests/data/tiny/data_dir/smoothed/1.csv") as smoothed:
        assert smoothed.read() == "42,0.5\n50,1.0\n52,0.0\n"
    with open("tests/data/tiny/data_dir/smoothed/2.csv") as smoothed:
        assert smoothed.read() == "1000,0.0\n1234,1.0\n1235,1.0\n"
    shutil.rmtree("tests/data/tiny/data_dir/smoothed/")
    assert result.exit_code == 0, result.output


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scan_cli():
    """
    pretty poor test currently, since the test data set is not useful.
    it just makes sure that the CLI doesn't crash.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["scan", "--threads", "1", "--bandwidth", "2", "tests/data/tiny/data_dir_smooth/", "-"])
    assert result.exit_code == 0, result.output
    assert "Found no variable regions" in result.output


def test_matrix_cli(tmp_path):
    outfile = os.path.join(tmp_path, "mtx.csv")
    bed = "1\t50\t52\tx\n2\t1000\t1234\ty\n"
    runner = CliRunner()
    result = runner.invoke(cli, ["matrix", "--keep-other-columns", "-", "tests/data/tiny/data_dir_smooth/", outfile], input=bed)
    mtx = read_csv(outfile)
    assert result.exit_code == 0, result.output
    assert mtx["meth_frac"].values.tolist() == [.5, .0, .5, .5]
    assert mtx["bed_col4"].values.tolist() == ["x", "x", "y", "y"]


def test_profile_cli():
    bed = "1\t51\t51\t+\tx\n2\t1234\t1234\t-\ty\n"
    runner = CliRunner()
    result = runner.invoke(cli, ["profile", "--strand-column", "4", "--width", "2", "-", "tests/data/tiny/data_dir/", "-"], input=bed)
    profile = ("position,cell,n_meth,cell_name,n_total,meth_frac,ci_lower,ci_upper\n"
               "-1,1,2,a,2,1.0,0.2902272522159686,1.0\n"
               "-1,2,1,b,1,1.0,0.167499485479413,1.0\n")
    assert result.exit_code == 0, result.output
    assert profile in result.output


def test_filter_cli_threshold(tmp_path):
    p = os.path.join(tmp_path, "filtered_data_dir")
    runner = CliRunner()
    result = runner.invoke(cli, ["filter", "--min-meth", "50", "tests/data/tiny/data_dir/", p])
    assert result.exit_code == 0, result.output
    with open(os.path.join(p, "column_header.txt")) as colnames:
        assert colnames.read().strip() == "b"
    with open(os.path.join(p, "cell_stats.csv")) as csv:
        assert csv.readline().startswith("cell_name,")
        assert csv.readline().startswith("b,")


def test_filter_cli_keep(tmp_path):
    p = os.path.join(tmp_path, "filtered_data_dir")
    keep_txt = os.path.join(tmp_path, "cells_to_keep.txt")
    with open(keep_txt, "w") as f:
        f.write("a\n\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["filter", "--cell-names", keep_txt, "tests/data/tiny/data_dir/", p])
    assert result.exit_code == 0, result.output
    with open(os.path.join(p, "column_header.txt")) as colnames:
        assert colnames.read().strip() == "a"
    with open(os.path.join(p, "cell_stats.csv")) as csv:
        assert csv.readline().startswith("cell_name,")
        assert csv.readline().startswith("a,")


def test_filter_cli_discard(tmp_path):
    p = os.path.join(tmp_path, "filtered_data_dir")
    discard_txt = os.path.join(tmp_path, "cells_to_discard.txt")
    with open(discard_txt, "w") as f:
        f.write("\na\n\na\n\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["filter", "--discard", "--cell-names", discard_txt, "tests/data/tiny/data_dir/", p])
    assert result.exit_code == 0, result.output
    with open(os.path.join(p, "column_header.txt")) as colnames:
        assert colnames.read().strip() == "b"
    with open(os.path.join(p, "cell_stats.csv")) as csv:
        assert csv.readline().startswith("cell_name,")
        assert csv.readline().startswith("b,")
