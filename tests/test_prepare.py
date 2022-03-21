import os
import scipy.sparse as sp_sparse
from click.testing import CliRunner
from scbs.prepare import _get_cell_names, _human_to_computer
from scbs.cli import cli


class MockFile():
    def __init__(self, name):
        self.name = name


def test_get_cell_name():
    f = [MockFile("dir/a.csv"), MockFile("/dir/dir2/b.csv.gz")]
    assert _get_cell_names(f) == ["a", "b"]


def test_coverage_format_creation():
    assert _human_to_computer("bismark") == (0, 1, 4, 5, False, "\t", False,)
    assert _human_to_computer("allc") == (0, 1, 4, 5, True, "\t", True,)
    assert _human_to_computer("1:2:3:4c:\\t:1") == (0, 1, 2, 3, True, "\t", True,)
    assert _human_to_computer("1:2:3:4m:\\t:1") == (0, 1, 2, 3, False, "\t", True,)
    assert _human_to_computer("1:2:3:4m:\\t:0") == (0, 1, 2, 3, False, "\t", False,)


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
