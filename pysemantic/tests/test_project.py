#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""Tests for the project class."""

import os.path as op
import os
import tempfile
import shutil
import warnings
import unittest
from ConfigParser import RawConfigParser, NoSectionError
from copy import deepcopy

import pandas as pd
import numpy as np
import yaml
from pandas.io.parsers import ParserWarning

import pysemantic.project as pr
from pysemantic.tests.test_base import (BaseProjectTestCase, TEST_DATA_DICT,
                                        TEST_CONFIG_FILE_PATH, _dummy_postproc)
from pysemantic.errors import MissingProject
from pysemantic.utils import colnames

try:
    from yaml import CLoader as Loader
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader
    from yaml import Dumper

try:
    import tables
    PYTABLES_NOT_INSTALLED = False
except ImportError:
    PYTABLES_NOT_INSTALLED = True

try:
    import xlrd
    XLRD_NOT_INSTALLED = False
except ImportError:
    XLRD_NOT_INSTALLED = True

try:
    import openpyxl
    OPENPYXL_NOT_INSTALLED = False
except ImportError:
    OPENPYXL_NOT_INSTALLED = True


class TestProjectModule(BaseProjectTestCase):

    """Tests for the project module level functions."""

    def test_get_datasets(self):
        """Test the get_datasets function returns the correct datasets."""
        datasets = pr.get_datasets("pysemantic")
        ideal = ['person_activity', 'multi_iris', 'iris', 'bad_iris',
                'random_row_iris']
        self.assertItemsEqual(ideal, datasets)

    def test_get_datasets_no_project(self):
        """Test if the get_datasets function works with no project name."""
        dataset_names = pr.get_datasets()
        self.assertTrue("pysemantic" in dataset_names)
        ideal = ['person_activity', 'multi_iris', 'iris', 'bad_iris',
                'random_row_iris']
        self.assertItemsEqual(dataset_names['pysemantic'], ideal)

    def test_add_dataset(self):
        """Test if adding datasets programmatically works fine."""
        tempdir = tempfile.mkdtemp()
        outpath = op.join(tempdir, "foo.csv")
        dframe = pd.DataFrame(np.random.random((10, 10)))
        dframe.to_csv(outpath, index=False)
        specs = dict(path=outpath, delimiter=',', nrows=10)
        try:
            pr.add_dataset("pysemantic", "sample_dataset", specs)
            parsed_specs = pr.get_schema_specs("pysemantic", "sample_dataset")
            self.assertKwargsEqual(specs, parsed_specs)
        finally:
            shutil.rmtree(tempdir)
            with open(TEST_DATA_DICT, "r") as fileobj:
                test_specs = yaml.load(fileobj, Loader=Loader)
            del test_specs['sample_dataset']
            with open(TEST_DATA_DICT, "w") as fileobj:
                yaml.dump(test_specs, fileobj, Dumper=Dumper,
                          default_flow_style=False)

    def test_remove_dataset(self):
        """Test if programmatically removing a dataset works."""
        with open(TEST_DATA_DICT, "r") as fileobj:
            specs = yaml.load(fileobj, Loader=Loader)
        try:
            pr.remove_dataset("pysemantic", "iris")
            self.assertRaises(KeyError, pr.get_schema_specs, "pysemantic",
                              "iris")
        finally:
            with open(TEST_DATA_DICT, "w") as fileobj:
                yaml.dump(specs, fileobj, Dumper=Dumper,
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

    def test_add_project_relpath(self):
        """Check that adding a project with relative path to schemafile fails."""
        self.assertRaises(ValueError, pr.add_project, "test_relpath", "foo/bar")

    def test_remove_project(self):
        """Test if removing a project works properly."""
        self.assertTrue(pr.remove_project("test_project"))
        self.assertRaises(NoSectionError, pr.get_default_specfile,
                          "test_project")


class TestProjectClass(BaseProjectTestCase):

    """Tests for the project class and its methods."""

    def test_min_dropna_on_cols(self):
        """Test if specifying a minimum value for a column also drops the NaNs."""
        x1 = np.random.rand(10, 2) / 10
        x2 = np.random.rand(10, 2)
        x = np.r_[x1, x2]
        np.random.shuffle(x)
        df = pd.DataFrame(x, columns="col_a col_b".split())
        df.loc[3, "col_a"] = np.nan
        df.loc[7, "col_b"] = np.nan
        tempdir = tempfile.mkdtemp()
        data_dir = op.join(tempdir, "data")
        os.mkdir(data_dir)
        schema_fpath = op.join(tempdir, "schema.yml")
        data_fpath = op.join(data_dir, "data.csv")
        df.to_csv(data_fpath, index=False)
        schema = {'data': {'path': op.join("data", "data.csv"),
            "column_rules": {"col_a": {"min": 0.1}}}}
        with open(schema_fpath, "w") as fin:
            yaml.dump(schema, fin, Dumper=Dumper, default_flow_style=False)
        pr.add_project("min_dropna_col", schema_fpath)
        try:
            loaded = pr.Project("min_dropna_col").load_dataset("data")
            self.assertFalse(np.any(pd.isnull(loaded)))
            self.assertGreater(loaded['col_a'].min(), 0.1)
        finally:
            pr.remove_project("min_dropna_col")
            shutil.rmtree(tempdir)

    def test_relpath(self):
        """Test if specifying datapaths relative to schema workds."""
        df = pd.DataFrame(np.random.randint(low=1, high=10, size=(10, 2)),
                          columns="a b".split())
        tempdir = tempfile.mkdtemp()
        data_dir = op.join(tempdir, "data")
        os.mkdir(data_dir)
        schema_fpath = op.join(tempdir, "schema.yml")
        data_fpath = op.join(data_dir, "data.csv")
        df.to_csv(data_fpath, index=False)
        schema = {'data': {'path': op.join("data", "data.csv"),
                           "dataframe_rules": {"drop_duplicates": False}}}
        with open(schema_fpath, "w") as fin:
            yaml.dump(schema, fin, Dumper=Dumper, default_flow_style=False)
        pr.add_project("relpath", schema_fpath)
        try:
            loaded = pr.Project("relpath").load_dataset("data")
            self.assertDataFrameEqual(loaded, df)
        finally:
            pr.remove_project("relpath")
            shutil.rmtree(tempdir)

    def test_nrows_shuffling(self):
        """test_relpath"""
        """Test if the shuffle parameter works with the nrows parameter."""
        tempdir = tempfile.mkdtemp()
        schema_fpath = op.join(tempdir, "schema.yml")
        data_fpath = op.join(tempdir, "data.csv")
        X = np.c_[np.arange(10), np.arange(10)]
        ix = range(5) + "a b c d e".split()
        df = pd.DataFrame(X, index=ix)
        df.to_csv(data_fpath, index_label="index")
        schema = {'data': {'path': data_fpath, "index_col": "index",
                           'nrows': {'count': 5, "shuffle": True}}}
        with open(schema_fpath, "w") as fin:
            yaml.dump(schema, fin, Dumper=Dumper, default_flow_style=False)
        pr.add_project("nrows_shuffle", schema_fpath)
        try:
            df = pr.Project("nrows_shuffle").load_dataset("data")
            for row_label in "a b c d e".split():
                self.assertNotIn(row_label, df.index)
            self.assertFalse(np.all(df.index == range(5)))
        finally:
            pr.remove_project("nrows_shuffle")
            shutil.rmtree(tempdir)

    def test_index_column_exclude(self):
        """Test if values are excluded from index column if so specified."""
        tempdir = tempfile.mkdtemp()
        schema_fpath = op.join(tempdir, "schema.yml")
        data_fpath = op.join(tempdir, "data.csv")
        df = pd.DataFrame.from_dict({'index': np.arange(10), 'col_a':
                                     np.arange(10)})
        df.to_csv(data_fpath, index=False)
        schema = {'data': {'path': data_fpath, 'index_col': 'index',
                           'column_rules': {'index': {'exclude': [1, 2]}}}}
        with open(schema_fpath, "w") as fin:
            yaml.dump(schema, fin, Dumper=Dumper, default_flow_style=False)
        pr.add_project("index_exclude", schema_fpath)
        try:
            df = pr.Project("index_exclude").load_dataset("data")
            self.assertItemsEqual(df.shape, (8, 1))
            self.assertEqual(df.index.name, "index")
            self.assertNotIn(1, df.index)
            self.assertNotIn(2, df.index)
        finally:
            pr.remove_project("index_exclude")
            shutil.rmtree(tempdir)

    def test_index_column_rules(self):
        """Test if column rules specified for index columns are enforced."""
        schema = {'iris': {'path': self.data_specs['iris']['path'],
                           'index_col': 'Species',
                           'dataframe_rules': {'drop_duplicates': False},
                           'column_rules': {'Species': {'regex': '.*e.*'}}}}
        with tempfile.NamedTemporaryFile(delete=False) as f_schema:
            yaml.dump(schema, f_schema, Dumper=Dumper, default_flow_style=False)
        pr.add_project("index_col_rules", f_schema.name)
        try:
            project = pr.Project("index_col_rules")
            df = project.load_dataset("iris")
            self.assertEqual(df.index.name.lower(), 'species')
            self.assertNotIn("virginica", df.index.unique())
        finally:
            pr.remove_project("index_col_rules")
            os.unlink(f_schema.name)

    def test_indexcol_not_in_usecols(self):
        """
        Test if the specified index column is added to the usecols
        argument."""
        schema = {'iris': {'path': self.data_specs['iris']['path'],
                           'index_col': 'Species',
                           'use_columns': ['Sepal Length', 'Petal Width']}}
        with tempfile.NamedTemporaryFile(delete=False) as f_schema:
            yaml.dump(schema, f_schema, Dumper=Dumper, default_flow_style=False)
        pr.add_project("testindex_usecols", f_schema.name)
        try:
            project = pr.Project("testindex_usecols")
            df = project.load_dataset("iris")
            self.assertEqual(df.index.name, "Species")
            self.assertItemsEqual(df.columns, ['Sepal Length', 'Petal Width'])
        finally:
            pr.remove_project("testindex_usecols")
            os.unlink(f_schema.name)

    def test_invalid_literals(self):
        """Test if columns containing invalid literals are parsed safely."""
        tempdir = tempfile.mkdtemp()
        schema_fpath = op.join(tempdir, "schema.yml")
        data_fpath = op.join(tempdir, "data.csv")
        data = pd.DataFrame.from_dict(dict(col_a=range(10)))
        data['col_b'] = ["x"] * 10
        data.to_csv(data_fpath, index=False)
        schema = {'dataset': {'path': data_fpath, 'dtypes': {'col_a': int,
                                                             'col_b': int}}}
        with open(schema_fpath, "w") as fin:
            yaml.dump(schema, fin, Dumper=Dumper, default_flow_style=False)
        pr.add_project("invalid_literal", schema_fpath)
        try:
            pr.Project("invalid_literal").load_dataset('dataset')
        finally:
            shutil.rmtree(tempdir)
            pr.remove_project("invalid_literal")

    def test_index_col(self):
        """Test if specifying the index_col works."""
        iris_fpath = self.expected_specs['iris']['filepath_or_buffer']
        specs = {'path': iris_fpath, 'index_col': 'Species',
                 'dataframe_rules': {'drop_duplicates': False}}
        pr.add_dataset("pysemantic", "iris_indexed", specs)
        try:
            df = pr.Project('pysemantic').load_dataset('iris_indexed')
            for specie in ['setosa', 'versicolor', 'virginica']:
                self.assertEqual(df.ix[specie].shape[0], 50)
        finally:
            pr.remove_dataset('pysemantic', 'iris_indexed')

    def test_multiindex(self):
        """Test if providing a list of indices in the schema returns a proper
        multiindexed dataframe."""
        pa_fpath = self.expected_specs['person_activity']['filepath_or_buffer']
        index_cols = ['sequence_name', 'tag']
        specs = {'path': pa_fpath, 'index_col': index_cols, 'delimiter': '\t'}
        pr.add_dataset("pysemantic", "pa_multiindex", specs)
        try:
            df = pr.Project('pysemantic').load_dataset('pa_multiindex')
            self.assertTrue(isinstance(df.index, pd.MultiIndex))
            self.assertEqual(len(df.index.levels), 2)
            seq_name, tags = df.index.levels
            org_df = pd.read_table(specs['path'])
            for col in index_cols:
                x = org_df[col].unique().tolist()
                y = df.index.get_level_values(col).unique().tolist()
                self.assertItemsEqual(x, y)
        finally:
            pr.remove_dataset('pysemantic', 'pa_multiindex')

    @unittest.skipIf(OPENPYXL_NOT_INSTALLED, "Loading Excel files requires openpyxl.")
    def test_load_excel_multisheet(self):
        """Test combining multiple sheets into a single dataframe."""
        tempdir = tempfile.mkdtemp()
        spreadsheet = op.join(tempdir, "multifile_iris.xlsx")
        iris = self.project.load_dataset("iris")
        with pd.ExcelWriter(spreadsheet) as writer:
            iris.to_excel(writer, "iris1", index=False)
            iris.to_excel(writer, "iris2", index=False)
        schema = {'iris': {'path': spreadsheet, 'sheetname': ['iris1', 'iris2'],
                           'dataframe_rules': {'drop_duplicates': False}}}
        schema_fpath = op.join(tempdir, "multi_iris.yaml")
        with open(schema_fpath, "w") as fout:
            yaml.dump(schema, fout, Dumper=Dumper, default_flow_style=False)
        pr.add_project("multi_iris", schema_fpath)
        try:
            ideal = pd.concat((iris, iris), axis=0)
            actual = pr.Project('multi_iris').load_dataset("iris")
            self.assertDataFrameEqual(ideal, actual)
        finally:
            pr.remove_project("multi_iris")
            shutil.rmtree(tempdir)

    @unittest.skipIf(XLRD_NOT_INSTALLED, "Reading Excel files requires xlrd.")
    def test_load_excel_sheetname(self):
        """Test if specifying the sheetname loads the correct dataframe."""
        xl_project = pr.Project("test_excel")
        ideal_iris = self.project.load_dataset("iris")
        actual_iris = xl_project.load_dataset("iris_renamed")
        self.assertDataFrameEqual(ideal_iris, actual_iris)

    @unittest.skipIf(XLRD_NOT_INSTALLED, "Reading Excel files requires xlrd.")
    def test_load_excel(self):
        """Test if excel spreadsheets are read properly from the schema."""
        xl_project = pr.Project("test_excel")
        ideal_iris = self.project.load_dataset("iris")
        actual_iris = xl_project.load_dataset("iris")
        self.assertDataFrameEqual(ideal_iris, actual_iris)

    def test_nrows_callable(self):
        """Check if specifying the nrows argument as a callable works."""
        nrows = lambda x: np.remainder(x, 2) == 0
        iris_specs = pr.get_schema_specs("pysemantic", "iris")
        iris_specs['nrows'] = nrows
        project = pr.Project(schema={'iris': iris_specs})
        loaded = project.load_dataset('iris')
        self.assertEqual(loaded.shape[0], 75)
        ideal_ix = np.arange(150, step=2)
        np.testing.assert_allclose(ideal_ix, loaded.index.values)

    def test_random_row_selection_within_range(self):
        """Check if randomly selecting rows within a range works."""
        iris_specs = pr.get_schema_specs("pysemantic", "iris")
        iris_specs['nrows'] = {'range': [25, 75], 'count': 10, 'random': True}
        iris_specs['header'] = 0
        del iris_specs['dtypes']
        iris_specs['column_names'] = colnames(iris_specs['path'])
        project = pr.Project(schema={'iris': iris_specs})
        loaded = project.load_dataset('iris')
        self.assertEqual(loaded.shape[0], 10)
        ix = loaded.index.values
        self.assertTrue(ix.max() <= 50)

    def test_row_selection_range(self):
        """Check if a range of rows can be selected from the dataset."""
        iris_specs = pr.get_schema_specs("pysemantic", "iris")
        iris_specs['nrows'] = {'range': [25, 75]}
        iris_specs['header'] = 0
        del iris_specs['dtypes']
        iris_specs['column_names'] = colnames(iris_specs['path'])
        project = pr.Project(schema={'iris': iris_specs})
        loaded = project.load_dataset('iris')
        self.assertEqual(loaded.shape[0], 50)
        ideal_ix = np.arange(50)
        self.assertTrue(np.allclose(loaded.index.values, ideal_ix))

    def test_row_selection_random_range(self):
        """Check if a range of rows can be selected from the dataset."""
        iris_specs = pr.get_schema_specs("pysemantic", "iris")
        iris_specs['nrows'] = {'range': [25, 75], 'random': True}
        iris_specs['header'] = 0
        del iris_specs['dtypes']
        iris_specs['column_names'] = colnames(iris_specs['path'])
        project = pr.Project(schema={'iris': iris_specs})
        loaded = project.load_dataset('iris')
        self.assertEqual(loaded.shape[0], 50)
        ideal_ix = np.arange(50)
        self.assertFalse(np.all(loaded.index.values == ideal_ix))

    def test_random_row_directive(self):
        """Check if the schema for randomizing rows works."""
        loaded = self.project.load_dataset("random_row_iris")
        self.assertEqual(loaded.shape[0], 50)
        ideal_ix = np.arange(50)
        self.assertFalse(np.all(loaded.index.values == ideal_ix))

    def test_random_row_selection(self):
        iris_specs = pr.get_schema_specs("pysemantic", "iris")
        iris_specs['nrows'] = dict(random=True, count=50)
        project = pr.Project(schema={'iris': iris_specs})
        loaded = project.load_dataset('iris')
        self.assertEqual(loaded.shape[0], 50)
        ideal_ix = np.arange(50)
        self.assertFalse(np.all(loaded.index.values == ideal_ix))

    def test_export_dataset_csv(self):
        """Test if the default csv exporter works."""
        tempdir = tempfile.mkdtemp()
        project = pr.Project("pysemantic")
        try:
            dataset = "iris"
            outpath = op.join(tempdir, dataset + ".csv")
            project.export_dataset(dataset, outpath=outpath)
            self.assertTrue(op.exists(outpath))
            loaded = pd.read_csv(outpath)
            self.assertDataFrameEqual(loaded, project.load_dataset(dataset))
        finally:
            shutil.rmtree(tempdir)

    def test_exclude_cols(self):
        """Test if importing data with excluded columns works."""
        filepath = op.join(op.abspath(op.dirname(__file__)), "testdata",
                           "iris.csv")
        specs = {'path': filepath, 'exclude_columns': ['Species']}
        pr.add_dataset('pysemantic', 'excl_iris', specs)
        try:
            project = pr.Project("pysemantic")
            loaded = project.load_dataset("excl_iris")
            self.assertNotIn('Species', loaded.columns)
        finally:
            pr.remove_dataset("pysemantic", "excl_iris")

    def test_column_postprocessors(self):
        """Test if postprocessors work on column data properly."""
        filepath = op.join(op.abspath(op.dirname(__file__)), "testdata",
                           "iris.csv")
        col_rules = {'Species': {'postprocessors': [_dummy_postproc]}}
        specs = {'path': filepath, 'column_rules': col_rules}
        pr.add_dataset("pysemantic", "postproc_iris", specs)
        try:
            project = pr.Project("pysemantic")
            loaded = project.load_dataset("postproc_iris")
            processed = loaded['Species']
            self.assertNotIn("setosa", processed.unique())
        finally:
            pr.remove_dataset("pysemantic", "postproc_iris")

    def test_na_reps(self):
        """Test if the NA representations are parsed properly."""
        project = pr.Project("pysemantic")
        loaded = project.load_dataset("bad_iris")
        self.assertItemsEqual(loaded.shape, (300, 5))

    def test_na_reps_list(self):
        """Test if NA values work when specified as a list."""
        tempdir = tempfile.mkdtemp()
        df = pd.DataFrame(np.random.rand(10, 2))
        ix = np.random.randint(0, df.shape[0], size=(5,))
        ix = np.unique(ix)
        df.iloc[ix, 0] = "foo"
        df.iloc[ix, 1] = "bar"
        fpath = op.join(tempdir, "test_na.csv")
        df.to_csv(fpath, index=False)
        schema = {'path': fpath, 'na_values': ["foo", "bar"],
                  'dataframe_rules': {'drop_na': False,
                                      'drop_duplicates': False}}
        schema_fpath = op.join(tempdir, "test_na.yaml")
        with open(schema_fpath, "w") as fid:
            yaml.dump({'test_na': schema}, fid, Dumper=Dumper,
                      default_flow_style=False)
        pr.add_project("test_na", schema_fpath)
        try:
            df = pr.Project("test_na").load_dataset("test_na")
            self.assertEqual(pd.isnull(df).sum().sum(), ix.shape[0] * 2)
        finally:
            pr.remove_project("test_na")
            shutil.rmtree(tempdir)

    def test_global_na_reps(self):
        """Test is specifying a global NA value for a dataset works."""
        tempdir = tempfile.mkdtemp()
        df = pd.DataFrame(np.random.rand(10, 10))
        ix = np.random.randint(0, df.shape[0], size=(5,))
        ix = np.unique(ix)
        for i in xrange(ix.shape[0]):
            df.iloc[ix[i], ix[i]] = "foobar"
        fpath = op.join(tempdir, "test_na.csv")
        df.to_csv(fpath, index=False)
        schema = {'path': fpath, 'na_values': "foobar",
                  'dataframe_rules': {'drop_na': False,
                                      'drop_duplicates': False}}
        schema_fpath = op.join(tempdir, "test_na.yaml")
        with open(schema_fpath, "w") as fid:
            yaml.dump({'test_na': schema}, fid, Dumper=Dumper,
                      default_flow_style=False)
        pr.add_project("test_na", schema_fpath)
        try:
            df = pr.Project("test_na").load_dataset("test_na")
            self.assertEqual(pd.isnull(df).sum().sum(), ix.shape[0])
        finally:
            pr.remove_project("test_na")
            shutil.rmtree(tempdir)

    def test_error_bad_lines_correction(self):
        """test if the correction for bad lines works."""
        tempdir = tempfile.mkdtemp()
        iris_path = op.join(op.abspath(op.dirname(__file__)), "testdata",
                            "iris.csv")
        with open(iris_path, "r") as fid:
            iris_lines = fid.readlines()
        outpath = op.join(tempdir, "bad_iris.csv")
        iris_lines[50] = iris_lines[50].rstrip() + ",0,23,\n"
        with open(outpath, 'w') as fid:
            fid.writelines(iris_lines)
        data_dict = op.join(tempdir, "dummy_project.yaml")
        specs = {'bad_iris': {'path': outpath}}
        with open(data_dict, "w") as fid:
            yaml.dump(specs, fid, Dumper=Dumper, default_flow_style=False)
        pr.add_project('dummy_project', data_dict)
        try:
            project = pr.Project('dummy_project')
            df = project.load_dataset('bad_iris')
            self.assertItemsEqual(df.shape, (147, 5))
        finally:
            shutil.rmtree(tempdir)
            pr.remove_project('dummy_project')

    @unittest.skipIf(PYTABLES_NOT_INSTALLED, "HDF export requires PyTables.")
    def test_export_dataset_hdf(self):
        """Test if exporting the dataset to hdf works."""
        tempdir = tempfile.mkdtemp()
        project = pr.Project("pysemantic")
        try:
            for dataset in project.datasets:
                if dataset not in ("bad_iris", "random_row_iris"):
                    outpath = op.join(tempdir, dataset + ".h5")
                    project.export_dataset(dataset, outpath=outpath)
                    self.assertTrue(op.exists(outpath))
                    group = r'/{0}/{1}'.format(project.project_name, dataset)
                    loaded = pd.read_hdf(outpath, group)
                    self.assertDataFrameEqual(loaded,
                                              project.load_dataset(dataset))
        finally:
            shutil.rmtree(tempdir)

    def test_reload_data_dict(self):
        """Test if the reload_data_dict method works."""
        project = pr.Project("pysemantic")
        tempdir = tempfile.mkdtemp()
        datapath = op.join(tempdir, "data.csv")
        ideal = pd.DataFrame(np.random.randint(0, 9, size=(10, 5)),
                             columns=map(str, range(5)))
        ideal.to_csv(datapath, index=False)
        with open(TEST_DATA_DICT, "r") as fid:
            specs = yaml.load(fid, Loader=Loader)
        specs['fakedata'] = dict(path=datapath)
        with open(TEST_DATA_DICT, "w") as fid:
            yaml.dump(specs, fid, Dumper=Dumper)
        try:
            project.reload_data_dict()
            actual = project.load_dataset("fakedata")
            self.assertDataFrameEqual(ideal, actual)
        finally:
            shutil.rmtree(tempdir)
            del specs['fakedata']
            with open(TEST_DATA_DICT, "w") as fid:
                yaml.dump(specs, fid, Dumper=Dumper)

    def test_update_dataset(self):
        """Test if the update_dataset method works."""
        tempdir = tempfile.mkdtemp()
        _pr = pr.Project("pysemantic")
        iris = _pr.load_dataset("iris")
        x = np.random.random((150,))
        y = np.random.random((150,))
        iris['x'] = x
        iris['y'] = y
        org_cols = iris.columns.tolist()
        outpath = op.join(tempdir, "iris.csv")
        with open(TEST_DATA_DICT, "r") as fid:
            org_specs = yaml.load(fid, Loader=Loader)
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
                yaml.dump(org_specs, fid, Dumper=Dumper,
                          default_flow_style=False)

    def test_update_dataset_deleted_columns(self):
        """Test if the update dataset method removes column specifications."""
        tempdir = tempfile.mkdtemp()
        _pr = pr.Project("pysemantic")
        iris = _pr.load_dataset("iris")
        outpath = op.join(tempdir, "iris.csv")
        with open(TEST_DATA_DICT, "r") as fid:
            org_specs = yaml.load(fid, Loader=Loader)
        try:
            del iris['Species']
            _pr.update_dataset("iris", iris, path=outpath)
            pr_reloaded = pr.Project("pysemantic")
            iris_reloaded = pr_reloaded.load_dataset("iris")
            self.assertNotIn("Species", iris_reloaded.columns)
            self.assertNotIn("Species", pr_reloaded.column_rules["iris"])
        finally:
            shutil.rmtree(tempdir)
            with open(TEST_DATA_DICT, "w") as fid:
                yaml.dump(org_specs, fid, Dumper=Dumper,
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
            with warnings.catch_warnings(record=True) as catcher:
                _pr = pr.Project("pysemantic")
                dframe = _pr.load_dataset("sample_dataset")
                assert len(catcher) == 3
                for i in range(3):
                    assert issubclass(catcher[i].category, ParserWarning)
            data.remove("col1")
            self.assertItemsEqual(map(int, data), dframe['col1'].tolist())
        finally:
            pr.remove_dataset("pysemantic", "sample_dataset")
            shutil.rmtree(tempdir)

    def test_load_dataset_wrong_dtypes_in_spec(self):
        """Test if the Loader can safely load columns that have a wrongly
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
            yaml.dump({'testdata': specs}, fileobj, Dumper=Dumper,
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
        """Test if the Loader can load columns with integers and NAs.

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
            yaml.dump({'testdata': specs}, fileobj, Dumper=Dumper,
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
            org_specs = yaml.load(fileobj, Loader=Loader)
        new_specs = deepcopy(org_specs)
        for dataset_specs in new_specs.itervalues():
            if "nrows" in dataset_specs:
                del dataset_specs['nrows']
        with open(TEST_DATA_DICT, "w") as fileobj:
            yaml.dump(new_specs, fileobj, Dumper=Dumper,
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
                yaml.dump(org_specs, fileobj, Dumper=Dumper,
                          default_flow_style=False)

    def test_get_project_specs(self):
        """Test if the project manager gets all specifications correctly."""
        specs = self.project.get_project_specs()
        del specs['bad_iris']
        del specs['random_row_iris']
        del specs['multi_iris']
        for name, argdict in specs.iteritems():
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
        for argset in outargs:
            argset['usecols'] = colnames(argset['filepath_or_buffer'])
        self.assertTrue(isinstance(outargs, list))
        self.assertEqual(len(outargs), len(self.expected_specs['multi_iris']))
        for i in range(len(outargs)):
            self.assertKwargsEqual(outargs[i],
                                   self.expected_specs['multi_iris'][i])

    def test_load_all(self):
        """Test if loading all datasets in a project works as expected."""
        loaded = self.project.load_datasets()
        self.assertItemsEqual(loaded.keys(), ('iris', 'person_activity',
                                              'multi_iris', 'bad_iris',
                                              'random_row_iris'))
        dframe = pd.read_csv(**self.expected_specs['iris'])
        self.assertDataFrameEqual(loaded['iris'], dframe)
        dframe = pd.read_csv(**self.expected_specs['person_activity'])
        self.assertDataFrameEqual(loaded['person_activity'], dframe)
        dframes = [pd.read_csv(**args) for args in
               self.expected_specs['multi_iris']]
        dframes = [x.drop_duplicates() for x in dframes]
        dframe = pd.concat(dframes)
        dframe.set_index(np.arange(dframe.shape[0]), inplace=True)
        self.assertDataFrameEqual(loaded['multi_iris'], dframe)

    def test_init_project_yaml_dump(self):
        """Test initialization of Project class with the raw yaml dump."""
        project_specs = pr.get_schema_specs('pysemantic')
        project = pr.Project(schema=project_specs)
        loaded = project.load_datasets()
        self.assertItemsEqual(loaded.keys(), ('iris', 'person_activity',
                                              'multi_iris', 'bad_iris',
                                              'random_row_iris'))
        dframe = pd.read_csv(**self.expected_specs['iris'])
        self.assertDataFrameEqual(loaded['iris'], dframe)
        dframe = pd.read_csv(**self.expected_specs['person_activity'])
        self.assertDataFrameEqual(loaded['person_activity'], dframe)
        dframes = [pd.read_csv(**args) for args in
               self.expected_specs['multi_iris']]
        dframes = [x.drop_duplicates() for x in dframes]
        dframe = pd.concat(dframes)
        dframe.set_index(np.arange(dframe.shape[0]), inplace=True)
        self.assertDataFrameEqual(loaded['multi_iris'], dframe)

    def test_dataset_colnames(self):
        """Check if the column names read by the Loader are correct."""
        for name, sep in {'iris': ',', 'person_activity': '\t'}.iteritems():
            loaded = self.project.load_dataset(name)
            columns = loaded.columns.tolist()
            spec_colnames = colnames(self.data_specs[name]['path'], sep=sep)
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
                    self.assertIn(colname, self.data_specs[name]['parse_dates'])
                else:
                    self.assertEqual(loaded[colname].dtype,
                                     self.data_specs[name]['dtypes'][colname])

if __name__ == '__main__':
    unittest.main()
