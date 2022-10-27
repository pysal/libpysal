#!/usr/bin/env python3

import os
import platform
import unittest
import numpy as np
import pandas

from .. import available
from ..base import get_data_home

os_name = platform.system()

class Testexamples(unittest.TestCase):

    def test_available(self):
        examples = available()
        self.assertEqual(type(examples), pandas.core.frame.DataFrame)

    def test_data_home(self): 
        pth = get_data_home()
        head, tail = os.path.split(pth)
        self.assertEqual(tail, 'pysal')
        if os_name == 'Linux':
            heads = head.split("/")
            self.assertEqual(heads[1], 'home')
            self.assertEqual(heads[-1], 'share')
            self.assertEqual(heads[-2], '.local')
        elif os_name == 'Darwin':
            heads = head.split("/")
            self.assertEqual(heads[1], 'Users')
            self.assertEqual(heads[-1], 'Application Support')
            self.assertEqual(heads[-2], 'Library')
        elif os_name == 'Windows':
            heads = head.split("\\")
            self.assertEqual(heads[1], 'Users')
            self.assertEqual(heads[-1], 'Local')
            self.assertEqual(heads[-2], 'AppData')

suite = unittest.TestLoader().loadTestsFromTestCase(Testexamples)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
