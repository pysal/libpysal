#!/usr/bin/env python3
import errno
import os
import platform
import tempfile
import unittest
import pandas
from unittest.mock import MagicMock, patch

from platformdirs import user_data_dir

from .. import available, get_url, load_example

from ..base import get_data_home

os_name = platform.system()

original_path_exists = os.path.exists
original_makedirs = os.makedirs



class Testexamples(unittest.TestCase):
    def test_available(self):
        examples = available()
        self.assertEqual(type(examples), pandas.core.frame.DataFrame)
        self.assertEqual(examples.shape, (98, 3))

    def test_data_home(self):
        pth = get_data_home()
        head, tail = os.path.split(pth)
        self.assertEqual(tail, "pysal")
        if os_name == "Linux":
            if "XDG_DATA_HOME" in os.environ:
                self.assertEqual(head, os.environ["XDG_DATA_HOME"])
            else:
                heads = head.split("/")
                self.assertEqual(heads[-1], "share")
                self.assertEqual(heads[-2], ".local")
        elif os_name == "Darwin":
            heads = head.split("/")
            self.assertEqual(heads[1], "Users")
            self.assertEqual(heads[-1], "Application Support")
            self.assertEqual(heads[-2], "Library")
        elif os_name == "Windows":
            heads = head.split("\\")
            self.assertEqual(heads[1], "Users")
            self.assertEqual(heads[-2], "Local")
            self.assertEqual(heads[-3], "AppData")

    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_data_home_fallback(self, path_exists_mock, makedirs_mock):
        data_home = user_data_dir("pysal", "pysal")

        def makedirs_side_effect(path, exist_ok=False):
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

        self.assertEqual(tail, "pysal")
        self.assertEqual(head, tempfile.gettempdir())


    def test_get_url(self):
        self.assertEqual(get_url("10740"), None)
        url = "https://geodacenter.github.io/data-and-lab//data/baltimore.zip"
        self.assertEqual(get_url("Baltimore"), url)

    def test_load_example(self):
        taz = load_example("taz")
        flist = taz.get_file_list()
        self.assertEqual(len(flist), 4)


suite = unittest.TestLoader().loadTestsFromTestCase(Testexamples)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
