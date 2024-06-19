#!/usr/bin/env python3
import errno
import os
import platform
import tempfile
from unittest.mock import patch

import pandas
from platformdirs import user_data_dir

from .. import available, get_url, load_example
from ..base import get_data_home

os_name = platform.system()

original_path_exists = os.path.exists
original_makedirs = os.makedirs


class TestExamples:
    def test_available(self):
        examples = available()
        assert type(examples) == pandas.core.frame.DataFrame
        assert examples.shape == (99, 3)

    def test_data_home(self):
        pth = get_data_home()
        head, tail = os.path.split(pth)
        assert tail == "pysal"
        if os_name == "Linux":
            if "XDG_DATA_HOME" in os.environ:
                assert head == os.environ["XDG_DATA_HOME"]
            else:
                heads = head.split("/")
                assert heads[-1] == "share"
                assert heads[-2] == ".local"
        elif os_name == "Darwin":
            heads = head.split("/")
            assert heads[1] == "Users"
            assert heads[-1] == "Application Support"
            assert heads[-2] == "Library"
        elif os_name == "Windows":
            heads = head.split("\\")
            assert heads[1] == "Users"
            assert heads[-2] == "Local"
            assert heads[-3] == "AppData"

    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_data_home_fallback(self, path_exists_mock, makedirs_mock):
        data_home = user_data_dir("pysal", "pysal")

        def makedirs_side_effect(path, exist_ok=False):  # noqa: ARG001
            if path == data_home:
                raise OSError(errno.EROFS)

        def path_exists_side_effect(path):
            if path == data_home:
                return False
            return original_path_exists(path)

        makedirs_mock.side_effect = makedirs_side_effect
        path_exists_mock.side_effect = path_exists_side_effect

        pth = get_data_home()
        head, tail = os.path.split(pth)

        assert tail == "pysal"
        assert head == tempfile.gettempdir()

    def test_get_url(self):
        assert get_url("10740") is None
        url = "https://geodacenter.github.io/data-and-lab//data/baltimore.zip"
        assert get_url("Baltimore") == url

    def test_load_example(self):
        taz = load_example("taz")
        flist = taz.get_file_list()
        assert len(flist) == 4
