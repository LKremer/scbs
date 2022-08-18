import os
import shutil

import pytest
from click.testing import CliRunner
from pandas import read_csv

from scbs.cli import cli


def test_diff_cli(tmp_path):
    outfile = os.path.join(tmp_path, "dmr.bed")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "diff",
            "tests/data/tiny_diff/",
            "tests/data/tiny_diff/celltypes.txt",
            "--min-cells",
            "3",
            outfile,
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        "Determined threshold of 2.4871385120811462 for neuroblast of real data."
        in result.output
    )
    dmr = read_csv(outfile, sep="\t", header=None)
    assert dmr[0].sum() == 5324
    assert dmr[1].sum() == 31250980767
    assert dmr[2].sum() == 31252558767
    assert dmr[3].sum() == -86.47359063078119
    assert dmr[5].sum() == 186.0982029998757
    assert len(dmr[dmr[4] == "neuroblast"]) == 284
    assert dmr[0][493] == 1
    assert dmr[2][135] == 69439700
    assert dmr[4][237] == "neuroblast"


def test_smooth_cli():
    runner = CliRunner()
    result = runner.invoke(
        cli, ["smooth", "--bandwidth", "2", "tests/data/tiny/data_dir/"]
    )
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
    result = runner.invoke(
        cli,
        [
            "scan",
            "--threads",
            "1",
            "--bandwidth",
            "2",
            "tests/data/tiny/data_dir_smooth/",
            "-",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Found no variable regions" in result.output


def test_matrix_cli(tmp_path):
    outfile = os.path.join(tmp_path, "mtx.csv")
    bed = "1\t50\t52\tx\n2\t1000\t1234\ty\n"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "matrix",
            "--keep-other-columns",
            "-",
            "tests/data/tiny/data_dir_smooth/",
            outfile,
        ],
        input=bed,
    )
    mtx = read_csv(outfile)
    assert result.exit_code == 0, result.output
    assert mtx["meth_frac"].values.tolist() == [0.5, 0.0, 0.5, 0.5]
    assert mtx["bed_col4"].values.tolist() == ["x", "x", "y", "y"]


def test_profile_cli():
    bed = "1\t51\t51\t+\tx\n2\t1234\t1234\t-\ty\n"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "profile",
            "--strand-column",
            "4",
            "--width",
            "2",
            "-",
            "tests/data/tiny/data_dir/",
            "-",
        ],
        input=bed,
    )
    profile = (
        "position,cell,n_meth,cell_name,n_total,meth_frac,ci_lower,ci_upper\n"
        "-1,1,2,a,2,1.0,0.2902272522159686,1.0\n"
        "-1,2,1,b,1,1.0,0.167499485479413,1.0\n"
    )
    assert result.exit_code == 0, result.output
    assert profile in result.output


def test_filter_cli_threshold(tmp_path):
    p = os.path.join(tmp_path, "filtered_data_dir")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["filter", "--min-meth", "50", "tests/data/tiny/data_dir/", p]
    )
    assert result.exit_code == 0, result.output
    with open(os.path.join(p, "column_header.txt")) as colnames:
        assert colnames.read().strip() == "b"
    with open(os.path.join(p, "cell_stats.csv")) as csv:
        assert csv.readline().startswith("cell_name,")
        assert csv.readline().startswith("b,")


def test_filter_cli_toostrict(tmp_path):
    p = os.path.join(tmp_path, "filtered_data_dir")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["filter", "--min-meth", "100", "tests/data/tiny/data_dir/", p]
    )
    assert result.exit_code == 1, result.output


def test_filter_cli_keep(tmp_path):
    p = os.path.join(tmp_path, "filtered_data_dir")
    keep_txt = os.path.join(tmp_path, "cells_to_keep.txt")
    with open(keep_txt, "w") as f:
        f.write("a\n\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["filter", "--cell-names", keep_txt, "tests/data/tiny/data_dir/", p]
    )
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
    result = runner.invoke(
        cli,
        [
            "filter",
            "--discard",
            "--cell-names",
            discard_txt,
            "tests/data/tiny/data_dir/",
            p,
        ],
    )
    assert result.exit_code == 0, result.output
    with open(os.path.join(p, "column_header.txt")) as colnames:
        assert colnames.read().strip() == "b"
    with open(os.path.join(p, "cell_stats.csv")) as csv:
        assert csv.readline().startswith("cell_name,")
        assert csv.readline().startswith("b,")
