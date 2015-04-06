#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""Tests for the project class."""

import os.path as op
import tempfile
import shutil
import warnings
import datetime
import unittest
from ConfigParser import RawConfigParser, NoSectionError
from copy import deepcopy

import pandas as pd
import numpy as np
import yaml
from pandas.io.parsers import ParserWarning

import pysemantic.project as pr
from pysemantic.tests.test_base import (BaseProjectTestCase, TEST_DATA_DICT,
                                        TEST_CONFIG_FILE_PATH)
from pysemantic.errors import MissingProject


class TestProjectModule(BaseProjectTestCase):

    """Tests for the project module level functions."""

    def test_get_datasets(self):
        """Test the get_datasets function returns the correct datasets."""
        datasets = pr.get_datasets("pysemantic")
        ideal = ['person_activity', 'multi_iris', 'iris']
        self.assertItemsEqual(ideal, datasets)

    def test_get_datasets_no_project(self):
        """Test if the get_datasets function works with no project name."""
        dataset_names = pr.get_datasets()
        self.assertTrue("pysemantic" in dataset_names)
        ideal = ['person_activity', 'multi_iris', 'iris']
        self.assertItemsEqual(dataset_names['pysemantic'], ideal)

    def test_add_dataset(self):
        """Test if adding datasets programmatically works fine."""
        tempdir = tempfile.mkdtemp()
        outpath = op.join(tempdir, "foo.csv")
        dframe = pd.DataFrame(np.random.random((10, 10)))
        dframe.to_csv(outpath, index=False)
        specs = dict(path=outpath, delimiter=',', ncols=10, nrows=10)
        try:
            pr.add_dataset("pysemantic", "sample_dataset", specs)
            parsed_specs = pr.get_schema_specs("pysemantic", "sample_dataset")
            self.assertKwargsEqual(specs, parsed_specs)
        finally:
            shutil.rmtree(tempdir)
            with open(TEST_DATA_DICT, "r") as fileobj:
                test_specs = yaml.load(fileobj, Loader=yaml.CLoader)
            del test_specs['sample_dataset']
            with open(TEST_DATA_DICT, "w") as fileobj:
                yaml.dump(test_specs, fileobj, Dumper=yaml.CDumper,
                          default_flow_style=False)

    def test_remove_dataset(self):
        """Test if programmatically removing a dataset works."""
        with open(TEST_DATA_DICT, "r") as fileobj:
            specs = yaml.load(fileobj, Loader=yaml.CLoader)
        try:
            pr.remove_dataset("pysemantic", "iris")
            self.assertRaises(KeyError, pr.get_schema_specs, "pysemantic",
                              "iris")
        finally:
            with open(TEST_DATA_DICT, "w") as fileobj:
                yaml.dump(specs, fileobj, Dumper=yaml.CDumper,
                          default_flow_style=False)

    def test_get_schema_spec(self):
        """Test the module level function to get schema specifications."""
        specs = pr.get_schema_specs("pysemantic")
        self.assertKwargsEqual(specs, self.data_specs)

    def test_set_schema_fpath(self):
        """Test if programmatically setting a schema file to an existing
        project works."""
        old_schempath = pr.get_default_specfile("pysemantic")
        try:
            self.assertTrue(pr.set_schema_fpath("pysemantic", "/foo/bar"))
            self.assertEqual(pr.get_default_specfile("pysemantic"),
                             "/foo/bar")
            self.assertRaises(MissingProject, pr.set_schema_fpath,
                              "foobar", "/foo/bar")
        finally:
            conf_path = pr.locate_config_file()
            parser = RawConfigParser()
            parser.read(conf_path)
            parser.remove_option("pysemantic", "specfile")
            parser.set("pysemantic", "specfile", old_schempath)
            with open(TEST_CONFIG_FILE_PATH, "w") as fileobj:
                parser.write(fileobj)

    def test_add_project(self):
        """Test if adding a project works properly."""
        test_project_name = "test_project"
        pr.add_project(test_project_name, TEST_DATA_DICT)
        # Check if the project name is indeed present in the config file
        test_dict = pr.get_default_specfile(test_project_name)
        self.assertTrue(test_dict, TEST_DATA_DICT)

    def test_remove_project(self):
        """Test if removing a project works properly."""
        self.assertTrue(pr.remove_project("test_project"))
        self.assertRaises(NoSectionError, pr.get_default_specfile,
                          "test_project")


class TestProjectClass(BaseProjectTestCase):

    """Tests for the project class and its methods."""

    def test_update_dataset(self):
        tempdir = tempfile.mkdtemp()
        """Test if the update_dataset method works."""
        _pr = pr.Project("pysemantic")
        iris = _pr.load_dataset("iris")
        x = np.random.random((150,))
        y = np.random.random((150,))
        iris['x'] = x
        iris['y'] = y
        org_cols = iris.columns.tolist()
        outpath = op.join(tempdir, "iris.csv")
        with open(TEST_DATA_DICT, "r") as fid:
            org_specs = yaml.load(fid, Loader=yaml.CLoader)
        try:
            _pr.update_dataset("iris", iris, path=outpath, sep='\t')
            _pr = pr.Project("pysemantic")
            iris = _pr.load_dataset("iris")
            self.assertItemsEqual(org_cols, iris.columns.tolist())
            iris_validator = _pr.validators['iris']
            updated_args = iris_validator.parser_args
            self.assertEqual(updated_args['dtype']['x'], float)
            self.assertEqual(updated_args['dtype']['y'], float)
            self.assertEqual(updated_args['sep'], '\t')
            self.assertEqual(updated_args['filepath_or_buffer'], outpath)
        finally:
            shutil.rmtree(tempdir)
            with open(TEST_DATA_DICT, "w") as fid:
                yaml.dump(org_specs, fid, Dumper=yaml.CDumper,
                          default_flow_style=False)

    def test_regex_separator(self):
        """Test if the project properly loads a dataset when it encounters
        regex separators.
        """
        tempdir = tempfile.mkdtemp()
        outfile = op.join(tempdir, "sample.txt")
        data = ["col1"] + map(str, range(10))
        with open(outfile, "w") as fileobj:
            fileobj.write("\n".join(data))
        specs = dict(path=outfile, delimiter=r'\n', dtypes={'col1': int})
        pr.add_dataset("pysemantic", "sample_dataset", specs)
        try:
            _pr = pr.Project("pysemantic")
            with warnings.catch_warnings(record=True) as catcher:
                dframe = _pr.load_dataset("sample_dataset")
                assert len(catcher) == 2
                assert issubclass(catcher[1].category, ParserWarning)
            data.remove("col1")
            self.assertItemsEqual(map(int, data), dframe['col1'].tolist())
        finally:
            pr.remove_dataset("pysemantic", "sample_dataset")
            shutil.rmtree(tempdir)

    def test_load_dataset_wrong_dtypes_in_spec(self):
        """Test if the loader can safely load columns that have a wrongly
        specified data type in the schema.
        """
        # Make a file with two columns, both specified as integers in the
        # dtypes, but one has random string types.
        x = np.random.randint(0, 10, size=(100, 2))
        dframe = pd.DataFrame(x, columns=['a', 'b'])
        tempdir = tempfile.mkdtemp()
        outfile = op.join(tempdir, "testdata.csv")
        _ix = np.random.randint(0, 100, size=(5,))
        dframe['b'][_ix] = "aa"
        dframe.to_csv(outfile, index=False)
        specs = dict(delimiter=',', dtypes={'a': int, 'b': int}, path=outfile)
        specfile = op.join(tempdir, "dict.yaml")
        with open(specfile, "w") as fileobj:
            yaml.dump({'testdata': specs}, fileobj, Dumper=yaml.CDumper,
                      default_flow_style=False)
        pr.add_project("wrong_dtype", specfile)
        try:
            _pr = pr.Project("wrong_dtype")
            with warnings.catch_warnings(record=True) as catcher:
                dframe = _pr.load_dataset("testdata")
                assert len(catcher) == 1
                assert issubclass(catcher[-1].category, UserWarning)
        finally:
            pr.remove_project("wrong_dtype")
            shutil.rmtree(tempdir)

    def test_integer_col_na_values(self):
        """Test if the loader can load columns with integers and NAs.

        This is necessary because NaNs cannot be represented by integers."""
        x = map(str, range(20))
        x[13] = ""
        df = pd.DataFrame.from_dict(dict(a=x, b=x))
        tempdir = tempfile.mkdtemp()
        outfile = op.join(tempdir, "testdata.csv")
        df.to_csv(outfile, index=False)
        specfile = op.join(tempdir, "dict.yaml")
        specs = dict(delimiter=',', dtypes={'a': int, 'b': int}, path=outfile)
        with open(specfile, "w") as fileobj:
            yaml.dump({'testdata': specs}, fileobj, Dumper=yaml.CDumper,
                      default_flow_style=False)
        pr.add_project("wrong_dtype", specfile)
        try:
            _pr = pr.Project("wrong_dtype")
            df = _pr.load_dataset("testdata")
            self.assertEqual(df['a'].dtype, float)
            self.assertEqual(df['b'].dtype, float)
        finally:
            pr.remove_project("wrong_dtype")
            shutil.rmtree(tempdir)

    def test_load_dataset_missing_nrows(self):
        """Test if the project loads datasets properly if the nrows parameter
        is not provided in the schema.
        """
        # Modify the schema to remove the nrows
        with open(TEST_DATA_DICT, "r") as fileobj:
            org_specs = yaml.load(fileobj, Loader=yaml.CLoader)
        new_specs = deepcopy(org_specs)
        for dataset_specs in new_specs.itervalues():
            del dataset_specs['nrows']
        with open(TEST_DATA_DICT, "w") as fileobj:
            yaml.dump(new_specs, fileobj, Dumper=yaml.CDumper,
                      default_flow_style=False)
        try:
            _pr = pr.Project("pysemantic")
            dframe = pd.read_csv(**self.expected_specs['iris'])
            loaded = _pr.load_dataset("iris")
            self.assertDataFrameEqual(dframe, loaded)
            dframe = pd.read_table(**self.expected_specs['person_activity'])
            loaded = _pr.load_dataset("person_activity")
            self.assertDataFrameEqual(loaded, dframe)
        finally:
            with open(TEST_DATA_DICT, "w") as fileobj:
                yaml.dump(org_specs, fileobj, Dumper=yaml.CDumper,
                          default_flow_style=False)

    def test_get_project_specs(self):
        """Test if the project manager gets all specifications correctly."""
        specs = self.project.get_project_specs()
        for name, argdict in specs.iteritems():
            if isinstance(argdict, list):
                for i in range(len(argdict)):
                    self.assertKwargsEqual(argdict[i],
                                           self.expected_specs[name][i])
            else:
                self.assertKwargsEqual(argdict, self.expected_specs[name])

    def test_get_dataset_specs(self):
        """Check if the project manager produces specifications for each
        dataset correctly.
        """
        for name in ['iris', 'person_activity']:
            self.assertKwargsEqual(self.project.get_dataset_specs(name),
                                   self.expected_specs[name])

    def test_get_multifile_dataset_specs(self):
        """Test if the multifile dataset specifications are valid."""
        outargs = self.project.get_dataset_specs("multi_iris")
        self.assertTrue(isinstance(outargs, list))
        self.assertEqual(len(outargs), len(self.expected_specs['multi_iris']))
        for i in range(len(outargs)):
            self.assertKwargsEqual(outargs[i],
                                   self.expected_specs['multi_iris'][i])

    def test_set_dataset_specs(self):
        """Check if setting dataset specifications through the Project object
        works.
        """
        path = op.join(op.abspath(op.dirname(__file__)), "testdata",
                       "iris.csv")
        specs = dict(filepath_or_buffer=path,
                     usecols=['Sepal Length', 'Petal Width', 'Species'],
                     dtype={'Sepal Length': str})
        self.assertTrue(self.project.set_dataset_specs("iris", specs))
        expected = pd.read_csv(**specs)
        loaded = self.project.load_dataset("iris")
        self.assertDataFrameEqual(expected, loaded)

    def test_set_dataset_specs_to_file(self):
        """Check if newly set dataset specifications are written to file
        properly."""
        try:
            with open(TEST_DATA_DICT, "r") as fileobj:
                oldspecs = yaml.load(fileobj, Loader=yaml.CLoader)
            path = op.join(op.abspath(op.dirname(__file__)), "testdata",
                           "iris.csv")
            specs = dict(filepath_or_buffer=path,
                         usecols=['Sepal Length', 'Petal Width', 'Species'],
                         dtype={'Sepal Length': str})
            self.assertTrue(self.project.set_dataset_specs("iris", specs,
                                                           write_to_file=True))
            with open(TEST_DATA_DICT, "r") as fileobj:
                newspecs = yaml.load(fileobj, Loader=yaml.CLoader)
            self.assertKwargsEqual(newspecs['iris'], specs)
        finally:
            with open(TEST_DATA_DICT, "w") as fileobj:
                yaml.dump(oldspecs, fileobj, Dumper=yaml.CDumper,
                          default_flow_style=False)

    def test_load_all(self):
        """Test if loading all datasets in a project works as expected."""
        loaded = self.project.load_datasets()
        self.assertItemsEqual(loaded.keys(), ('iris', 'person_activity',
                                              'multi_iris'))
        dframe = pd.read_csv(**self.expected_specs['iris'])
        self.assertDataFrameEqual(loaded['iris'], dframe)
        dframe = pd.read_csv(**self.expected_specs['person_activity'])
        self.assertDataFrameEqual(loaded['person_activity'], dframe)
        dframes = [pd.read_csv(**args) for args in
               self.expected_specs['multi_iris']]
        dframes = [x.drop_duplicates() for x in dframes]
        dframe = pd.concat(dframes)
        self.assertDataFrameEqual(loaded['multi_iris'], dframe)

    def test_dataset_colnames(self):
        """Check if the column names read by the loader are correct."""
        for name in ['iris', 'person_activity']:
            loaded = self.project.load_dataset(name)
            columns = loaded.columns.tolist()
            spec_colnames = self.data_specs[name]['dtypes'].keys()
            self.assertItemsEqual(spec_colnames, columns)

    def test_dataset_coltypes(self):
        """Check whether the columns have the correct datatypes."""
        for name in ['iris', 'person_activity']:
            loaded = self.project.load_dataset(name)
            for colname in loaded:
                if loaded[colname].dtype == np.dtype('O'):
                    self.assertEqual(self.data_specs[name]['dtypes'][colname],
                                     str)
                elif loaded[colname].dtype == np.dtype('<M8[ns]'):
                    self.assertEqual(self.data_specs[name]['dtypes'][colname],
                                     datetime.date)
                else:
                    self.assertEqual(loaded[colname].dtype,
                                     self.data_specs[name]['dtypes'][colname])

if __name__ == '__main__':
    unittest.main()
