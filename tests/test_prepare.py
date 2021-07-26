from scbs.prepare import _get_cell_names, _human_to_computer


class MockFile():
    def __init__(self, name):
        self.name = name


def test_get_cell_name():
    good_files = [MockFile("dir/a.csv"), MockFile("dir/b.csv.gz")]
    x = _get_cell_names(good_files)
    assert ["a", "b"] == x


def test_coverage_format_creation():
    assert _human_to_computer("bismark") == (0, 1, 4, 5, False, "\t", False,)
    assert _human_to_computer("allc") == (0, 1, 4, 5, True, "\t", True,)
    assert _human_to_computer("1:2:3:4c:\\t:1") == (0, 1, 2, 3, True, "\t", True,)
    assert _human_to_computer("1:2:3:4m:\\t:1") == (0, 1, 2, 3, False, "\t", True,)
    assert _human_to_computer("1:2:3:4m:\\t:0") == (0, 1, 2, 3, False, "\t", False,)

