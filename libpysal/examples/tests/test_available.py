#!/usr/bin/env python3

import os
import unittest
import numpy as np
import pandas

from .. import available

class Testavailable(unittest.TestCase):

    def test_available(self):
        examples = available()
        self.assertEqual(type(examples), pandas.core.frame.DataFrame)


suite = unittest.TestLoader().loadTestsFromTestCase(Testavailable)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
