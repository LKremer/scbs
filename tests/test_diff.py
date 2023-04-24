import os

import numpy as np
import pandas as pd
from click.testing import CliRunner

from scbs.cli import cli
from scbs.diff import parse_cell_groups


def test_parse_cell_groups():
    group_df, groups = parse_cell_groups(
        "tests/data/tiny_diff/cell_groups.csv", "tests/data/tiny_diff/"
    )
    assert len(groups) == 2
    assert groups[0] == "neuroblast"
    assert groups[1] == "oligodendrocyte"
    assert set(np.where(group_df == "oligodendrocyte")[0]) == {0, 2, 6, 7, 9}
    assert set(np.where(group_df == "neuroblast")[0]) == {1, 3, 4, 5, 8}
    assert set(np.where(group_df == "-")[0]) == {10, 11}


def test_parse_cell_groups_alt():
    group_df, groups = parse_cell_groups(
        "tests/data/tiny_diff/cell_groups_alt.csv", "tests/data/tiny_diff/"
    )
    assert len(groups) == 2
    assert groups[0] == "neuroblast"
    assert groups[1] == "oligodendrocyte"
    assert set(np.where(group_df == "oligodendrocyte")[0]) == {0, 2, 6, 7, 9}
    assert set(np.where(group_df == "neuroblast")[0]) == {1, 3, 4, 5, 8}


def test_diff_cli(tmp_path):
    outfile = os.path.join(tmp_path, "dmr.bed")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "diff",
            "tests/data/tiny_diff/",
            "tests/data/tiny_diff/cell_groups.csv",
            "--min-cells",
            "3",
            outfile,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "threshold of 2.487" in result.output
    dmr = pd.read_csv(outfile, sep="\t", header=None)
    assert dmr[0].sum() == 5324
    assert dmr[1].sum() == 31250980767
    assert dmr[2].sum() == 31252060767
    assert dmr[3].sum() == -165.62208458348474
    assert dmr[11].sum() == 486.24726241344274
    assert len(dmr[dmr[9] == "neuroblast"]) == 284
    assert dmr[0][493] == 9
    assert dmr[2][135] == 58035292
    assert dmr[9][237] == "neuroblast"
    assert np.all(dmr[1] < dmr[2])
    assert np.all(dmr[4] > 0)
    assert np.all(dmr[5] >= 3)  # --min-cells
    assert np.all(dmr[6] >= 3)
    assert np.all(dmr[7] >= 0) and np.all(dmr[7] <= 1)
    assert np.all(dmr[8] >= 0) and np.all(dmr[8] <= 1)
