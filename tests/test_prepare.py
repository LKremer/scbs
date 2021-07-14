from scbs.prepare import _get_cell_names


class MockFile():
    def __init__(self, name):
        self.name = name


def test_get_cell_name():
    good_files = [MockFile("dir/a.csv"), MockFile("dir/b.csv.gz")]
    x = _get_cell_names(good_files)
    assert ["a", "b"] == x
