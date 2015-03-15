#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Tests
"""

import unittest
import yaml
from validator import DataDictValidator


class TestDataDictValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.specfile = "testdata/test_dictionary.yaml"
        with open(cls.specfile, "r") as f:
            cls.basespecs = yaml.load(f, Loader=yaml.CLoader)

    def test1(self):
        print self.basespecs


if __name__ == '__main__':
    unittest.main()
