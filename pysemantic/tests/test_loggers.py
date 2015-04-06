#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""Test if loggers are logging all events properly."""


import tempfile
import shutil
import os
import os.path as op
import unittest

import yaml

from pysemantic import project as pr
from pysemantic.tests.test_base import (BaseTestCase, TEST_CONFIG_FILE_PATH,
                                        _path_fixer, TEST_DATA_DICT)
from pysemantic import loggers


class TestLoggers(BaseTestCase):

    """Testcase for all logger events in the pysemantic module."""

    @classmethod
    def setUpClass(cls):
        cls.conf_file = op.join(os.getcwd(), "test.conf")
        shutil.copy(TEST_CONFIG_FILE_PATH, cls.conf_file)
        _path_fixer(cls.conf_file)
        with open(TEST_DATA_DICT, "r") as fid:
            cls.old_specs = yaml.load(fid, Loader=yaml.CLoader)
        _path_fixer(TEST_DATA_DICT)
        pr.CONF_FILE_NAME = "test.conf"

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.conf_file)
        with open(TEST_DATA_DICT, "w") as fid:
            yaml.dump(cls.old_specs, fid, Dumper=yaml.CDumper,
                      default_flow_style=False)

    def setUp(self):
        """1. Change log directory to somehere temporary.
        2.Create a project object, load some data."""
        self.logdir = tempfile.mkdtemp()
        loggers.LOGDIR = self.logdir
        self.project = pr.Project("pysemantic")

    def tearDown(self):
        shutil.rmtree(self.logdir)

    def test_log_location(self):
        """Check if the log files are created in the correct location."""
        self.project.load_dataset("iris")
        log_list = os.listdir(self.logdir)
        self.assertEqual(len(log_list), 1)
        self.assertTrue(log_list[0].startswith("pysemantic"))


if __name__ == '__main__':
    unittest.main()
