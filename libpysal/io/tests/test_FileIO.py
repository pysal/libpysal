# ruff: noqa: N999

from ...examples import get_path
from ..fileio import FileIO


def test_by_col_exists():
    """Test if the Metaclass is initializing and providing readers to its children."""

    fh1 = FileIO.open(get_path("columbus.dbf"))
    fh2 = FileIO.open(get_path("usjoin.csv"))

    assert hasattr(fh1, "by_col")
    assert hasattr(fh2, "by_col")
