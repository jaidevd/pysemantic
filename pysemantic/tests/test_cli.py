#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""Tests for the cli."""

import os
import shutil
import subprocess
import tempfile
import unittest
import os.path as op
from copy import deepcopy
from ConfigParser import RawConfigParser

import yaml
import pandas as pd
import numpy as np

from pysemantic.tests.test_base import (BaseTestCase, TEST_CONFIG_FILE_PATH,
                                        TEST_DATA_DICT)
from pysemantic import project as pr

try:
    from yaml import CLoader as Loader
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader as Loader
    from yaml import Dumper as Dumper

try:
    import tables
    PYTABLES_NOT_INSTALLED = False
except ImportError:
    PYTABLES_NOT_INSTALLED = True


class TestCLI(BaseTestCase):

    """Test the pysemantic CLI."""

    @classmethod
    def setUpClass(cls):
        os.environ['PYSEMANTIC_CONFIG'] = "test.conf"
        pr.CONF_FILE_NAME = "test.conf"
        cls.testenv = os.environ
        cls.test_config_path = op.join(os.getcwd(), "test.conf")
        shutil.copy(TEST_CONFIG_FILE_PATH, cls.test_config_path)
        # Change the relative paths in the config file to absolute paths
        parser = RawConfigParser()
        parser.read(cls.test_config_path)
        for section in parser.sections():
            schema_path = parser.get(section, "specfile")
            parser.remove_option(section, "specfile")
            parser.set(section, "specfile",
                       op.join(op.abspath(op.dirname(__file__)), schema_path))
        with open(cls.test_config_path, "w") as fileobj:
            parser.write(fileobj)
        # change the relative paths in the test dictionary to absolute paths
        with open(TEST_DATA_DICT, "r") as fileobj:
            cls.org_specs = yaml.load(fileobj, Loader=Loader)
        new_specs = deepcopy(cls.org_specs)
        for _, specs in new_specs.iteritems():
            path = specs['path']
            specs['path'] = op.join(op.abspath(op.dirname(__file__)), path)
        # Rewrite this to the file
        with open(TEST_DATA_DICT, "w") as fileobj:
            yaml.dump(new_specs, fileobj, Dumper=Dumper,
                      default_flow_style=False)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.test_config_path)
        # Rewrite the original specs back to the config dir
        with open(TEST_DATA_DICT, "w") as fileobj:
            yaml.dump(cls.org_specs, fileobj, Dumper=Dumper,
                      default_flow_style=False)

    def setUp(self):
        pr.add_project("dummy_project", "/foo/bar.yaml")

    def tearDown(self):
        pr.remove_project("dummy_project")

    def test_set_specification(self):
        """Test if the set-specs subcommand of the CLI worls properly."""
        org_specs = pr.get_schema_specs("pysemantic")
        cmd = ['semantic', 'set-specs', 'pysemantic', '--dataset', 'iris',
               '--dlm', '|']
        try:
            subprocess.check_call(cmd, env=self.testenv)
            new_specs = pr.get_schema_specs("pysemantic", "iris")
            self.assertEqual(new_specs['delimiter'], '|')
        finally:
            for dataset_name, specs in org_specs.iteritems():
                pr.set_schema_specs("pysemantic", dataset_name, **specs)

    def test_list_projects(self):
        """Test if the `list` subcommand of the CLI works properly."""
        cmd = ['semantic', 'list']
        output = subprocess.check_output(cmd, env=self.testenv).splitlines()
        path = op.join(op.abspath(op.dirname(__file__)),
                       "testdata/test_dictionary.yaml")
        excel_path = op.join(op.abspath(op.dirname(__file__)),
                       "testdata/test_excel.yaml")
        dummy_data = [("pysemantic", path), ("test_excel", excel_path),
                      ("dummy_project", "/foo/bar.yaml")]
        for i, config in enumerate(dummy_data):
            ideal = "Project {0} with specfile at {1}".format(*config)
            self.assertEqual(ideal, output[i])

    def test_list_datasets(self):
        """Test if the `list` subocmmand works for listing datasets."""
        command = "semantic list --project pysemantic"
        cmd = command.split(' ')
        datasets = pr.get_datasets("pysemantic")
        output = subprocess.check_output(cmd, env=self.testenv).splitlines()
        self.assertItemsEqual(datasets, output)

    def test_add(self):
        """Test if the `add` subcommand can add projects to the config file."""
        try:
            cmd = ['semantic', 'add', 'dummy_added_project', '/tmp/dummy.yaml']
            subprocess.check_call(cmd, env=self.testenv)
            projects = pr.get_projects()
            self.assertIn(("dummy_added_project", "/tmp/dummy.yaml"), projects)
        finally:
            pr.remove_project("dummy_added_project")

    def test_add_dataset(self):
        """Test if the add-dataset subcommand adds datasets to projects."""
        tempdir = tempfile.mkdtemp()
        outfile = op.join(tempdir, "testdata.csv")
        dframe = pd.DataFrame(np.random.random((10, 2)), columns=['a', 'b'])
        dframe.to_csv(outfile, index=False)
        cmd = ("semantic add-dataset testdata --project pysemantic --path {}"
               " --dlm ,")
        cmd = cmd.format(outfile).split(" ")
        try:
            subprocess.check_call(cmd, env=self.testenv)
            _pr = pr.Project("pysemantic")
            self.assertIn("testdata", _pr.datasets)
            specs = dict(path=outfile, delimiter=',')
            actual = pr.get_schema_specs("pysemantic", "testdata")
            self.assertKwargsEqual(specs, actual)
        finally:
            pr.remove_dataset("pysemantic", "testdata")
            shutil.rmtree(tempdir)

    def test_remove_dataset(self):
        """Test if removing datasets works from the command line."""
        # Add a temporary dataset and try to remove it.
        tempdir = tempfile.mkdtemp()
        outfile = op.join(tempdir, "testdata.csv")
        dframe = pd.DataFrame(np.random.random((10, 2)), columns=['a', 'b'])
        dframe.to_csv(outfile, index=False)
        specs = dict(path=outfile, delimiter=',')
        pr.add_dataset("pysemantic", "testdata", specs)
        try:
            command = "semantic remove pysemantic --dataset testdata"
            cmd = command.split(' ')
            subprocess.check_call(cmd, env=self.testenv)
            datasets = pr.get_datasets("pysemantic")
            self.assertNotIn("testdata", datasets)
        finally:
            datasets = pr.get_datasets("pysemantic")
            if "testdata" in datasets:
                pr.remove_dataset("pysemantic", "testdata")
            shutil.rmtree(tempdir)

    def test_remove(self):
        """Test if the remove subcommand can remove projects."""
        pr.add_project("dummy_project_2", "/foo/baz.yaml")
        try:
            cmd = ['semantic', 'remove', 'dummy_project_2']
            subprocess.check_call(cmd, env=self.testenv)
            projects = pr.get_projects()
            proj_names = [p[0] for p in projects]
            self.assertNotIn("dummy_project_2", proj_names)
        finally:
            pr.remove_project("dummy_project_2")

    def test_remove_nonexistent_project(self):
        """Check if attempting to remove a nonexistent project fails."""
        cmd = ['semantic', 'remove', 'foobar']
        output = subprocess.check_output(cmd, env=self.testenv)
        self.assertEqual(output.strip(), "Removing the project foobar failed.")

    def test_set_schema(self):
        """Test if the set-schema subcommand works."""
        cmd = ['semantic', 'set-schema', 'dummy_project', '/tmp/baz.yaml']
        subprocess.check_call(cmd, env=self.testenv)
        self.assertEqual(pr.get_default_specfile('dummy_project'),
                         '/tmp/baz.yaml')

    @unittest.skipIf(PYTABLES_NOT_INSTALLED, "HDF export needs PyTables.")
    def test_export_hdf(self):
        """Test if exporting a dataset to hdf works."""
        tempdir = tempfile.mkdtemp()
        cmd = "semantic export pysemantic --dataset iris {0}"
        cmd = cmd.format(op.join(tempdir, "iris.h5"))
        cmd = cmd.split()
        try:
            subprocess.check_call(cmd, env=self.testenv)
            self.assertTrue(op.exists(op.join(tempdir, "iris.h5")))
        finally:
            shutil.rmtree(tempdir)

    def test_set_schema_nonexistent_project(self):
        """Test if the set-schema prints proper warnings when trying to set
        schema file for nonexistent project.
        """
        cmd = ['semantic', 'set-schema', 'dummy_project_3', '/foo']
        output = subprocess.check_output(cmd, env=self.testenv)
        msg = """Project {} not found in the configuration. Please use
            $ semantic add
            to register the project.""".format("dummy_project_3")
        self.assertEqual(output.strip(), msg)

    def test_relative_path(self):
        """Check if the set-schema and add subcommands convert relative paths
        from the cmdline to absolute paths in the config file.
        """
        try:
            cmd = ['semantic', 'set-schema', 'dummy_project', './foo.yaml']
            subprocess.check_call(cmd, env=self.testenv)
            self.assertTrue(op.isabs(pr.get_default_specfile(
                                                             'dummy_project')))
            pr.remove_project("dummy_project")
            cmd = ['semantic', 'add', 'dummy_project', './foo.yaml']
            subprocess.check_call(cmd, env=self.testenv)
            self.assertTrue(op.isabs(pr.get_default_specfile(
                                                             'dummy_project')))
        finally:
            pr.remove_project("dummy_project_1")

if __name__ == '__main__':
    unittest.main()
