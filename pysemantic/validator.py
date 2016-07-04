#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""Traited Data validator for `pandas.DataFrame` objects."""

import copy
import cPickle
import json
import logging
import datetime
import textwrap
import warnings
import os.path as op
import yaml
import numpy as np
import pandas as pd
from pandas.io.parsers import ParserWarning
from pandas.parser import CParserError
from traits.api import (HasTraits, File, Property, Str, Dict, List, Type,
                        Bool, Either, push_exception_handler, cached_property,
                        Instance, Float, Any, TraitError)
from pysemantic.utils import TypeEncoder, get_md5_checksum, colnames
from pysemantic.custom_traits import AbsFile, ValidTraitList

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

push_exception_handler(lambda *args: None, reraise_exceptions=True)
logger = logging.getLogger(__name__)


class ParseErrorHandler(object):
    def __init__(self, parser_args, project, maxiter=None):
        self.parser_args = parser_args
        self.project = project
        fpath = self.parser_args.get('filepath_or_buffer',
                                     self.parser_args.get('io'))
        sep = self.parser_args.get('sep', False)
        if sep:
            self.colnames = colnames(fpath, sep=sep)
        else:
            self.colnames = colnames(fpath)
            self.parser_args['sep'] = sep
        if maxiter is not None:
            self.maxiter = maxiter
        else:
            self.maxiter = len(self.colnames)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def _update_parser(self, argdict):
        """Update the pandas parser based on the delimiter.

        :param argdict: Dictionary containing parser arguments.
        :return None:
        """
        fpath = argdict.get('filepath_or_buffer', argdict.get('io'))
        xls = fpath.endswith(".xlsx") or fpath.endswith("xls")
        if not self.project.user_specified_parser:
            if not xls:
                sep = argdict.get('sep', ",")
                if sep == ",":
                    self.parser = pd.read_csv
                else:
                    self.parser = pd.read_table
                    if sep == r'\t':
                        argdict.pop('sep', None)
            else:
                self.parser = self._load_excel_sheet

    def _load_excel_sheet(self, **parser_args):
        sheetname = parser_args.pop("sheetname")
        io = parser_args.pop('io')
        return pd.read_excel(io, sheetname=sheetname, **parser_args)

    def _update_dtypes(self, dtypes, typelist):
        """Update the dtypes parameter of the parser arguments.

        :param dtypes: The original column types
        :param typelist: List of tuples [(column_name, new_dtype), ...]
        """
        for colname, coltype in typelist:
            dtypes[colname] = coltype

    def _detect_row_with_na(self):
        """Return the list of columns in the dataframe, for which the data type
        has been marked as integer, but which contain NAs.

        :param parser_args: Dictionary containing parser arguments.
        """
        dtypes = self.parser_args.get("dtype")
        usecols = self.parser_args.get("usecols")
        if usecols is None:
            usecols = colnames(self.parser_args['filepath_or_buffer'])
        int_cols = [col for col in usecols if dtypes.get(col) is int]
        fpath = self.parser_args['filepath_or_buffer']
        sep = self.parser_args.get('sep', ',')
        nrows = self.parser_args.get('nrows')
        na_reps = {}
        if self.parser_args.get('na_values', False):
            for colname, na_vals in self.parser_args.get(
                    'na_values').iteritems():
                if colname in int_cols:
                    na_reps[colname] = na_vals
        converters = {}
        if self.parser_args.get('converters', False):
            for cname, cnv in self.parser_args.get('converters').iteritems():
                if cname in int_cols:
                    converters[cname] = cnv
        df = self.parser(fpath, sep=sep, usecols=int_cols, nrows=nrows,
                         na_values=na_reps, converters=converters)
        bad_rows = []
        for col in df:
            if np.any(pd.isnull(df[col])):
                bad_rows.append(col)
        return bad_rows

    def _detect_mismatched_dtype_row(self, specified_dtype):
        """Check the dataframe for rows that have a badly specified dtype.

        :param specfified_dtype: The datatype specified in the schema
        :param parser_args: Dictionary containing parser arguments.
        """
        to_read = []
        dtypes = self.parser_args.get("dtype")
        for key, value in dtypes.iteritems():
            if value is specified_dtype:
                to_read.append(key)
        fpath = self.parser_args['filepath_or_buffer']
        sep = self.parser_args.get('sep', ',')
        nrows = self.parser_args.get('nrows')
        df = self.parser(fpath, sep=sep, usecols=to_read, nrows=nrows,
                         error_bad_lines=False)
        bad_cols = []
        for col in df:
            try:
                df[col] = df[col].astype(specified_dtype)
            except ValueError:
                bad_cols.append(col)
                msg = textwrap.dedent("""\
                The specified dtype for the column '{0}' ({1}) seems to be
                incorrect. This has been ignored for now.
                Consider fixing this by editing the schema.""".format(col,
                                                                      specified_dtype))
                warnings.warn(msg, UserWarning)
        return bad_cols

    def _remove_unsafe_integer_columns(self, loc):
        bad_col = self.colnames[loc]
        del self.parser_args['dtype'][bad_col]

    def _detect_column_with_invalid_literals(self):
        dtypes = self.parser_args.pop('dtype')
        df = self.parser(**self.parser_args)
        bad_cols = []
        for colname, dtype in dtypes.iteritems():
            try:
                df[colname].astype(dtype)
            except (ValueError, TypeError):
                bad_cols.append(colname)
            except KeyError:
                if df.index.name == colname:
                    bad_cols.append(colname)
        self.parser_args['dtype'] = dtypes
        return bad_cols

    def load(self):
        """The main recursion loop."""
        self.c_iter = 0
        df = None
        while True:
            df = self._load()
            self.c_iter += 1
            if (self.c_iter > self.maxiter) or (df is not None):
                break
        return df

    def _load(self):
        """The actual loader function that does the heavy lifting.

        :param parser_args: Dictionary containing parser arguments.
        """
        self._update_parser(self.parser_args)
        try:
            return self.parser(**self.parser_args)
        except ValueError as e:
            if "Integer column has NA values" in e.message:
                bad_rows = self._detect_row_with_na()
                new_types = [(col, float) for col in bad_rows]
                self._update_dtypes(self.parser_args['dtype'], new_types)
                logger.info("Dtypes for following columns changed:")
                logger.info(json.dumps(new_types, cls=TypeEncoder))
                return self.parser(**self.parser_args)
            elif e.message.startswith("invalid literal"):
                bad_cols = self._detect_column_with_invalid_literals()
                msg = textwrap.dedent("""\
                        Columns {} designated as type integer could not
                        be safely cast as integers. Attempting to load as
                        string data. Consider changing the type in the schema.
                        """.format(bad_cols))
                logger.warn(msg)
                warnings.warn(msg, ParserWarning)
                for col in bad_cols:
                    del self.parser_args['dtype'][col]
                return self.parser(**self.parser_args)
            elif e.message.startswith("Falling back to the 'python' engine"):
                del self.parser_args['dtype']
                msg = textwrap.dedent("""\
                        Dtypes are not supported regex delimiters. Ignoring the
                        dtypes in the schema. Consider fixing this by editing
                        the schema for better performance.
                        """)
                logger.warn(msg)
                logger.info("Removing the dtype argument")
                warnings.warn(msg, ParserWarning)
                if "error_bad_lines" in self.parser_args:
                    del self.parser_args['error_bad_lines']
                return self.parser(**self.parser_args)
            elif e.message.startswith("cannot safely convert"):
                loc = int(e.message.split()[-1])
                bad_colname = self.colnames[loc]
                specified_dtype = self.parser_args['dtype'][bad_colname]
                self._remove_unsafe_integer_columns(loc)
                msg = textwrap.dedent("""\
                The specified dtype for the column '{0}' ({1}) seems to be
                incorrect. This has been ignored for now.
                Consider fixing this by editing the
                schema.""".format(bad_colname, specified_dtype))
                logger.warn(msg)
                logger.info("dtype for column {} removed.".format(bad_colname))
                warnings.warn(msg, UserWarning)
            elif e.message.startswith('could not convert string to float'):
                bad_cols = self._detect_mismatched_dtype_row(float)
                for col in bad_cols:
                    del self.parser_args['dtype'][col]
                msg = textwrap.dedent("""\
                The specified dtype for the column '{0}' ({1}) seems to be
                incorrect. This has been ignored for now.
                Consider fixing this by editing the schema.""".format(bad_cols,
                                                                      float))
                logger.warn(msg)
                logger.info("dtype removed for columns:".format(bad_cols))
                return self.parser(**self.parser_args)
        except AttributeError as e:
            if e.message == "'NoneType' object has no attribute 'dtype'":
                bad_rows = self._detect_mismatched_dtype_row(int)
                for col in bad_rows:
                    del self.parser_args['dtype'][col]
                logger.warn(msg)
                logger.info("dtype removed for columns:".format(bad_rows))
                return self.parser(**self.parser_args)
        except CParserError as e:
            self.parser_args['error_bad_lines'] = False
            msg = 'Adding the "error_bad_lines=False" argument to the ' + \
                  'list of parser arguments.'
            logger.info(msg)
            return self.parser(**self.parser_args)
        except Exception as e:
            if "Integer column has NA values" in e.message:
                bad_rows = self._detect_row_with_na()
                new_types = [(col, float) for col in bad_rows]
                self._update_dtypes(self.parser_args['dtype'], new_types)
                logger.info("Dtypes for following columns changed:")
                logger.info(json.dumps(new_types, cls=TypeEncoder))
                return self.parser(**self.parser_args)


class DataFrameValidator(HasTraits):
    """A validator class for `pandas.DataFrame` objects."""

    # The dataframe in question
    data = Instance(pd.DataFrame)

    # the column rules to be enforced
    column_rules = Dict

    # rules related to the dataset itself
    rules = Dict

    # Column to set as index
    index_col = Property(Any, depends_on=['rules'])

    # whether to drop duplicates
    is_drop_duplicates = Property(Bool, depends_on=['rules'])

    # whether to drop NAs
    is_drop_na = Property(Bool, depends_on=['rules'])

    # Names of columns to be rewritten
    column_names = Property(Any, depends_on=['rules'])

    # Specifications relating to the selection of rows.
    nrows = Property(Any, depends_on=['rules'])

    # Whether to shuffle the rows of the dataframe before returning
    shuffle = Property(Bool, depends_on=['rules'])

    # Unique values to maintain per column
    unique_values = Property(Dict, depends_on=['column_rules'])

    def _rules_default(self):
        return {}

    @cached_property
    def _get_shuffle(self):
        return self.rules.get("shuffle", False)

    @cached_property
    def _get_index_col(self):
        return self.rules.get('index_col', False)

    @cached_property
    def _get_nrows(self):
        return self.rules.get('nrows', {})

    @cached_property
    def _get_is_drop_na(self):
        return self.rules.get("drop_na", True)

    @cached_property
    def _get_is_drop_duplicates(self):
        return self.rules.get("drop_duplicates", True)

    @cached_property
    def _get_column_names(self):
        return self.rules.get("column_names")

    @cached_property
    def _get_unique_values(self):
        uvals = {}
        if self.column_rules is not None:
            for colname, rules in self.column_rules.iteritems():
                uvals[colname] = rules.get('unique_values', [])
        return uvals

    def apply_uniques(self):
        for colname, uniques in self.unique_values.iteritems():
            if colname in self.data:
                org_vals = self.data[colname].unique()
                for val in org_vals:
                    if len(uniques) > 0:
                        if val not in uniques:
                            drop_ix = self.data.index[
                                self.data[colname] == val]
                            self.data.drop(drop_ix, axis=0, inplace=True)

    def rename_columns(self):
        """Rename columns in dataframe as per the schema."""
        if self.column_names is not None:
            logger.info("Renaming columns as follows:")
            logger.info(json.dumps(self.column_names, cls=TypeEncoder))
            if isinstance(self.column_names, dict):
                for old_name, new_name in self.column_names.iteritems():
                    if old_name in self.data:
                        self.data[new_name] = self.data.pop(old_name)
            elif callable(self.column_names):
                columns = self.data.columns.copy()
                for old_name in columns:
                    new_name = self.column_names(old_name)
                    self.data[new_name] = self.data.pop(old_name)
            elif isinstance(self.column_names, list):
                self.data.columns = self.column_names

    def clean(self):
        """Return the converted dataframe after enforcing all rules."""

        self.apply_uniques()

        if isinstance(self.nrows, dict):
            if len(self.nrows) > 0:
                if self.nrows.get('random', False):
                    ix = self.data.index.values.copy()
                    np.random.shuffle(ix)
                    self.data = self.data.ix[ix]
                count = self.nrows.get('count', self.data.shape[0])
                self.data = self.data.ix[self.data.index[:count]]
        elif callable(self.nrows):
            ix = self.nrows(self.data.index)
            self.data = self.data.ix[self.data.index[ix]]

        if self.is_drop_na:
            x = self.data.shape[0]
            try:
                self.data.dropna(inplace=True)
            except TypeError:
                print "Cannot drop na."
            y = self.data.shape[0]
            logger.info("{0} rows containing NAs were dropped.".format(x - y))

        if self.is_drop_duplicates:
            x = self.data.shape[0]
            try:
                self.data.drop_duplicates(inplace=True)
            except TypeError:
                print "Cannot drop duplicate rows."
            y = self.data.shape[0]
            logger.info("{0} duplicate rows were dropped.".format(x - y))

        for col in self.data:
            logger.info("Commence cleaning of column {}".format(col))
            series = self.data[col]
            rules = self.column_rules.get(col, {})
            validator = SeriesValidator(data=series, rules=rules)
            self.data[col] = validator.clean()
            if len(validator.exclude_values) > 0:
                for exval in validator.exclude_values:
                    self.data.drop(self.data.index[self.data[col] == exval],
                                   inplace=True)
                logger.info("Excluding following values from col {0}".format(
                    col))
                logger.info(json.dumps(validator.exclude_values))
                # self.data.dropna(inplace=True)
        self.rename_columns()

        if self.index_col:
            self.data.set_index(self.index_col, drop=True, inplace=True)
            un_ix = self.data.index.unique()
            na_ix = pd.isnull(un_ix)
            self.data.drop(un_ix[na_ix], axis=0, inplace=True)

        if self.shuffle:
            self.data = self.data.sample(self.data.shape[0])

        return self.data


class SeriesValidator(HasTraits):
    """A validator class for `pandas.Series` objects."""

    # the series in question
    data = Instance(pd.Series)

    # Rules of validation
    rules = Dict

    # Whether to drop NAs from the series.
    is_drop_na = Property(Bool, depends_on=['rules'])

    # Whether to drop duplicates from the series.
    is_drop_duplicates = Property(Bool, depends_on=['rules'])

    # List of values to exclude
    exclude_values = Property(List, depends_on=['rules'])

    # Minimum value permitted in the series
    minimum = Property(Float, depends_on=['rules'])

    # Maximum value permitted in the series
    maximum = Property(Float, depends_on=['rules'])

    # Regular expression match for series containing strings
    regex = Property(Str, depends_on=['rules'])

    # List of postprocessors that work in the series
    postprocessors = Property(List, depends_on=['rules'])

    def do_postprocessing(self):
        for postprocessor in self.postprocessors:
            org_len = self.data.shape[0]
            logger.info("Applying postprocessor on column:")
            logger.info(json.dumps(postprocessor, cls=TypeEncoder))
            self.data = postprocessor(self.data)
            final_len = self.data.shape[0]
            if org_len != final_len:
                msg = ("Size of column changed after applying postprocessor."
                       "This could disturb the alignment of your data.")
                logger.warn(msg)
                warnings.warn(msg, UserWarning)

    def do_drop_duplicates(self):
        """Drop duplicates from the series if required."""
        if self.is_drop_duplicates:
            duplicates = self.data.index[self.data.duplicated()].tolist()
            logger.info("Following duplicated rows were dropped:")
            logger.info(json.dumps(duplicates))
            self.data.drop_duplicates(inplace=True)

    def do_drop_na(self):
        """Drop NAs from the series if required."""
        if self.is_drop_na:
            na_bool = pd.isnull(self.data)
            na_rows = self.data.index[na_bool].tolist()
            logger.info("Following rows containing NAs were dropped:")
            logger.info(json.dumps(na_rows))
            self.data.dropna(inplace=True)

    def drop_excluded(self):
        """Remove all values specified in `exclude_values`."""
        if len(self.exclude_values) > 0:
            logger.info("Removing the following exclude values:")
            logger.info(json.dumps(self.exclude_values))
            for value in self.exclude_values:
                self.data.drop(self.data.index[self.data == value],
                               inplace=True)

    def apply_minmax_rules(self):
        """Restrict the series to the minimum and maximum from the schema."""
        if self.data.dtype in (int, float, datetime.date):
            if self.minimum != -np.inf:
                logger.info("Setting minimum at {0}".format(self.minimum))
                self.data = self.data[self.data >= self.minimum]
            if self.maximum != np.inf:
                logger.info("Setting maximum at {0}".format(self.maximum))
                self.data = self.data[self.data <= self.maximum]

    def apply_regex(self):
        """Apply a regex filter on strings in the series."""
        if self.regex:
            if self.data.dtype is np.dtype('O'):
                # filter by regex
                logger.info("Applying regex filter with the following regex:")
                logger.info(self.regex)
                self.data = self.data[self.data.str.contains(self.regex)]

    def clean(self):
        """Return the converted series after enforcing all rules."""
        self.do_drop_duplicates()
        self.do_drop_na()
        self.do_postprocessing()
        self.apply_minmax_rules()
        self.apply_regex()
        return self.data

    @cached_property
    def _get_postprocessors(self):
        return self.rules.get("postprocessors", [])

    @cached_property
    def _get_exclude_values(self):
        return self.rules.get("exclude", [])

    @cached_property
    def _get_is_drop_na(self):
        return self.rules.get("drop_na", False)

    @cached_property
    def _get_is_drop_duplicates(self):
        return self.rules.get("drop_duplicates", False)

    @cached_property
    def _get_minimum(self):
        return self.rules.get("min", -np.inf)

    @cached_property
    def _get_maximum(self):
        return self.rules.get("max", np.inf)

    @cached_property
    def _get_regex(self):
        return self.rules.get("regex", "")


MYSQL_URL = "mysql+mysqldb://{username}:{password}@{hostname}/{db_name}"


class MySQLTableValidator(HasTraits):
    """A validator used when the data source is a mysql table."""

    # Specifications to use when making parser arguments
    specs = Dict

    # Name of the MySQL table to read
    table_name = Property(Str, depends_on=['specs'])

    # A dictionary containing the configuration
    config = Property(Dict, depends_on=['specs'])

    # Username used to connect to the DB
    username = Property(Str, depends_on=['config'])

    # Password used to connect to the DB
    password = Property(Str, depends_on=['config'])

    # hostname of the DB
    hostname = Property(Str, depends_on=['config'])

    # name of the database
    db_name = Property(Str, depends_on=['config'])

    # Chunksize
    chunksize = Property(Any, depends_on=['specs'])

    # Query
    query = Property(Str, depends_on=['specs'])

    # SQlAlchemy connection object to be used by the parser
    connection = Property(Any, depends_on=['username', 'password', 'hostname',
                                           'db_name'])

    # Parser args to be used by the pandas parser
    parser_args = Property(Dict, depends_on=['connection', 'specs'])

    @cached_property
    def _get_chunksize(self):
        return self.specs.get("chunksize")

    @cached_property
    def _get_config(self):
        return self.specs.get("config")

    @cached_property
    def _get_username(self):
        return self.config.get('username')

    @cached_property
    def _get_password(self):
        return self.config.get('password')

    @cached_property
    def _get_hostname(self):
        return self.config.get('hostname')

    @cached_property
    def _get_db_name(self):
        return self.config.get('db_name')

    @cached_property
    def _get_table_name(self):
        return self.config.get("table_name")

    @cached_property
    def _get_query(self):
        return self.specs.get("query")

    @cached_property
    def _get_connection(self):
        from sqlalchemy import create_engine
        url = MYSQL_URL.format(username=self.username, password=self.password,
                               hostname=self.hostname, db_name=self.db_name)
        return create_engine(url)

    @cached_property
    def _get_parser_args(self):
        return dict(table_name=self.table_name,
                    con=self.connection,
                    coerce_float=self.specs.get("coerce_float", True),
                    index_col=self.specs.get("index_col"),
                    parse_dates=self.specs.get("parse_dates"),
                    columns=self.specs.get("use_columns"),
                    chunksize=self.specs.get("chunksize"),
                    query=self.specs.get("query"))


POSTGRE_URL = "postgresql+psycopg2://{username}:{password}@{hostname}/{" \
              "db_name}"


class PostGRETableValidator(HasTraits):
    """A validator used when the data source is a postgres table."""

    # Specifications to use when making parser arguments
    specs = Dict

    # Name of the MySQL table to read
    table_name = Property(Str, depends_on=['specs'])

    # A dictionary containing the configuration
    config = Property(Dict, depends_on=['specs'])

    # Username used to connect to the DB
    username = Property(Str, depends_on=['config'])

    # Password used to connect to the DB
    password = Property(Str, depends_on=['config'])

    # hostname of the DB
    hostname = Property(Str, depends_on=['config'])

    # name of the database
    db_name = Property(Str, depends_on=['config'])

    # Query
    query = Property(Str, depends_on=['specs'])

    # Chunksize
    chunksize = Property(Any, depends_on=['specs'])

    # SQlAlchemy connection object to be used by the parser
    connection = Property(Any, depends_on=['username', 'password', 'hostname',
                                           'db_name'])

    # Parser args to be used by the pandas parser
    parser_args = Property(Dict, depends_on=['connection', 'specs'])

    @cached_property
    def _get_chunksize(self):
        return self.specs.get("chunksize")

    @cached_property
    def _get_config(self):
        return self.specs.get("config")

    @cached_property
    def _get_username(self):
        return self.config.get('username')

    @cached_property
    def _get_password(self):
        return self.config.get('password')

    @cached_property
    def _get_hostname(self):
        return self.config.get('hostname')

    @cached_property
    def _get_db_name(self):
        return self.config.get('db_name')

    @cached_property
    def _get_table_name(self):
        return self.specs.get("table_name")

    @cached_property
    def _get_connection(self):
        from sqlalchemy import create_engine
        url = POSTGRE_URL.format(username=self.username,
                                 password=self.password,
                                 hostname=self.hostname,
                                 db_name=self.db_name)
        return create_engine(url)

    @cached_property
    def _get_parser_args(self):
        return dict(table_name=self.table_name,
                    con=self.connection,
                    query=self.specs.get("query"),
                    coerce_float=self.specs.get("coerce_float", True),
                    index_col=self.specs.get("index_col"),
                    parse_dates=self.specs.get("parse_dates"),
                    columns=self.specs.get("use_columns"),
                    chunksize=self.specs.get("chunksize"))


TRAIT_NAME_MAP = {
    "filepath": "filepath_or_buffer",
    "nrows": "nrows",
    "index_col": "index_col",
    "delimiter": "sep",
    "dtypes": "dtype",
    "colnames": "usecols",
    "na_values": "na_values",
    "converters": "converters",
    "header": "header",
    "error_bad_lines": "error_bad_lines",
    "parse_dates": "parse_dates"
}


class SchemaValidator(HasTraits):
    """A validator class for schema in the data dictionary."""

    @classmethod
    def from_dict(cls, specification):
        """Get a validator from a schema dictionary.

        :param specification: Dictionary containing schema specifications.
        """
        return cls(specification=specification)

    @classmethod
    def from_specfile(cls, specfile, name, **kwargs):
        """Get a validator from a schema file.

        :param specfile: Path to the schema file.
        :param name: Name of the project to create the validator for.
        """
        return cls(specfile=specfile, name=name, **kwargs)

    def __init__(self, **kwargs):
        """Overwritten to ensure that the `required_args` trait is validated
        when the object is created, not when the trait is accessed.
        """
        super(SchemaValidator, self).__init__(**kwargs)
        if not kwargs.get('is_pickled', False):
            self.required_args = ['filepath', 'delimiter']

    # Public traits

    # Path to the data dictionary
    specfile = File(exists=True)

    # Name of the dataset described in the data dictionary
    name = Str

    # whether the data is a mysql table
    is_mysql = Property(Bool, depends_on=['specification'])

    # whether the data is a mysql table
    is_postgresql = Property(Bool, depends_on=['specification'])

    # Dict trait that holds the properties of the dataset
    specification = Dict

    # Path to the file containing the data
    filepath = Property(Either(AbsFile, List(AbsFile), Str),
                        depends_on=['specification', 'specfile'])

    # Whether the dataset spans multiple files
    is_multifile = Property(Bool, depends_on=['filepath'])

    # Whether the dataset is contained in a spreadsheet
    is_spreadsheet = Property(Bool, depends_on=['filepath'])

    # Default arguments for spreadsheets
    non_spreadsheet_args = List

    # Name of the sheet containing the dataframe. Only relevant when
    # is_spreadsheet is True
    sheetname = Property(Str, depends_on=['is_spreadsheet', 'specification'])

    # Delimiter
    delimiter = Property(Str, depends_on=['specification'])

    # number of rows in the dataset
    nrows = Property(Any, depends_on=['specification'])

    # Index column for the dataset
    index_col = Property(Any, depends_on=['specification'])

    # A dictionary whose keys are the names of the columns in the dataset, and
    # the keys are the datatypes of the corresponding columns
    dtypes = Dict(key_trait=Str, value_trait=Type)

    # Names of the columns in the dataset. This is just a convenience trait,
    # it's value is just a list of the keys of `dtypes`
    colnames = Property(List, depends_on=['specification', 'exclude_columns',
                                          'filepath', 'is_multifile'])

    # md5 checksum of the dataset file
    md5 = Property(Str, depends_on=['filepath'])

    # List of values that represent NAs
    na_values = Property(Any, depends_on=['specification'])

    # Default value of the `parse_dates` argument
    parse_dates = Property(Any, depends_on=['specification'])

    # List of converters to be applied to the columns. All converters are
    # assumed to be callables, which take the series as input and return a
    # series.
    converters = Property(Dict, depends_on=['specification'])

    # Header of the file
    header = Property(Any, depends_on=['specification'])

    # Names to use for columns in the dataframe
    column_names = Property(Any, depends_on=['specification'])

    # Rules for the dataframe that can only be enforeced after loading the
    # dataset, therefore must be exported to DataFrameValidator.
    df_rules = Dict

    # List of columns to exclude from the data
    exclude_columns = Property(List, depends_on=['specification'])

    # Whether pickled arguments exist in the schema
    is_pickled = Bool

    # Whether to raise errors on malformed lines
    error_bad_lines = Property(Bool, depends_on=['specification'])

    # Path to pickle file containing parser arguments
    pickle_file = Property(AbsFile, depends_on=['specification'])

    # Dictionary of arguments loaded from the pickle file.
    pickled_args = Property(Dict, depends_on=['pickle_file'])

    # List of required traits
    # FIXME: Arguments required by the schema should't have to be programmed
    # into the validator class. There must be a way to enforce requirements
    # right in the schema itself.
    required_args = ValidTraitList

    # Parser args for pandas
    parser_args = Property(Dict, depends_on=['filepath', 'delimiter', 'nrows',
                                             'dtypes', 'colnames'])

    # Protected traits

    _dtypes = Property(Dict(key_trait=Str, value_trait=Type),
                       depends_on=['specification'])

    # Public interface

    def get_parser_args(self):
        """Return parser args as required by pandas parsers."""
        return self.parser_args

    to_dict = get_parser_args

    def set_parser_args(self, specs, write_to_file=False):
        """Magic method required by Property traits."""
        self.parser_args = specs
        if write_to_file:
            logger.info("Following specs for dataset {0}".format(self.name) +
                        " were written to specfile {0}".format(self.specfile))
            with open(self.specfile, "r") as f:
                allspecs = yaml.load(f, Loader=Loader)
            allspecs[self.name] = specs
            with open(self.specfile, "w") as f:
                yaml.dump(allspecs, f, Dumper=Dumper,
                          default_flow_style=False)
        else:
            logger.info("Following parser args were set for dataset {}".format(
                self.name))
        logger.info(json.dumps(specs, cls=TypeEncoder))
        return True

    def _check_md5(self):
        import sys
        if sys.platform == 'win32':
            msg = "Verifying md5 checksums is not yet supported for your OS."
            logger.warn(msg)
            warnings.warn(msg, UserWarning)
            return
        if self.md5:
            if self.md5 != get_md5_checksum(self.filepath):
                msg = \
                    """The MD5 checksum of the file {} does not match the one
                     specified in the schema. This may not be the file you are
                     looking for."""
                logger.warn(msg.format(self.filepath))
                warnings.warn(msg.format(self.filepath), UserWarning)

    # Property getters and setters

    @cached_property
    def _get_is_mysql(self):
        if "source" in self.specification:
            return self.specification.get("source") == "mysql"
        return False

    @cached_property
    def _get_is_postgresql(self):
        if "source" in self.specification:
            return self.specification.get("source") == "postgresql"
        return False

    @cached_property
    def _get_parse_dates(self):
        parse_dates = self.specification.get("parse_dates", False)
        if parse_dates:
            if isinstance(parse_dates, str):
                parse_dates = [parse_dates]
        return parse_dates

    @cached_property
    def _get_filepath(self):
        if not self.is_pickled:
            fpath = self.specification.get('path', "")
        else:
            if not (self.is_mysql or self.is_postgresql):
                fpath = self.pickled_args['filepath_or_buffer']
            else:
                return ""
        if isinstance(fpath, list):
            for path in fpath:
                if not (op.exists(path) and op.isabs(path)):
                    raise TraitError("filepaths must be absolute.")
        elif isinstance(fpath, str):
            if not op.isabs(fpath):
                fpath = op.join(op.dirname(self.specfile), fpath)
            if not (self.is_mysql or self.is_postgresql):
                if not (op.exists(fpath) and op.isabs(fpath)):
                    raise TraitError("filepaths must be absolute.")
        return fpath

    @cached_property
    def _get_is_multifile(self):
        if not self.is_pickled:
            if isinstance(self.filepath, list):
                if len(self.filepath) > 1:
                    return True
        return False

    @cached_property
    def _get_is_spreadsheet(self):
        if (not self.is_multifile) and (not self.is_pickled):
            return self.filepath.endswith('.xls') or self.filepath.endswith(
                'xlsx')
        return False

    @cached_property
    def _get_index_col(self):
        ix_col = self.specification.get('index_col', None)
        if not isinstance(ix_col, list):
            if ix_col is not None:
                col_rules = self.specification.get("column_rules")
                if col_rules is not None:
                    if ix_col in col_rules:
                        self.df_rules["index_col"] = ix_col
                        return
        return ix_col

    @cached_property
    def _get_sheetname(self):
        if self.is_spreadsheet:
            return self.specification.get('sheetname', self.name)

    def _set_colnames(self, colnames):
        self.colnames = colnames

    @cached_property
    def _get_parser_args(self):
        if not (self.is_mysql or self.is_postgresql):
            self._check_md5()
            args = {}

            for traitname, argname in TRAIT_NAME_MAP.iteritems():
                args[argname] = getattr(self, traitname)

            # Date/Time arguments
            # FIXME: Allow for a mix of datetime column groupings and
            # individual
            # columns
            # All column renaming delegated to df_validtor
            if self.column_names is not None:
                self.df_rules['column_names'] = self.column_names

            if self.header not in (0, 'infer'):
                del args['usecols']

            if self.is_multifile:
                arglist = []
                for i in range(len(self.filepath)):
                    argset = copy.deepcopy(args)
                    argset.update({'filepath_or_buffer': self.filepath[i]})
                    argset.update({'nrows': self.nrows[i]})
                    arglist.append(argset)
                return arglist
            else:
                if self.filepath:
                    args.update({'filepath_or_buffer': self.filepath})
                if "nrows" in self.specification:
                    if isinstance(self.nrows, int):
                        args.update({'nrows': self.nrows})
                    elif isinstance(self.nrows, dict):
                        if self.nrows.get('random', False):
                            self.df_rules.update({'nrows': self.nrows})
                            del args['nrows']
                        if "range" in self.nrows:
                            start, stop = self.nrows['range']
                            args['skiprows'] = start
                            args['names'] = args.pop('usecols')
                            args['nrows'] = stop - start
                        if self.nrows.get("count", False) and \
                                self.nrows.get("shuffle", False):
                            args['nrows'] = self.nrows.get('count')
                            self.df_rules['shuffle'] = True
                    elif callable(self.nrows):
                        self.df_rules.update({'nrows': self.nrows})
                        del args['nrows']
                self.pickled_args.update(args)
                if self.is_spreadsheet:
                    self.pickled_args['sheetname'] = self.sheetname
                    self.pickled_args['io'] = self.pickled_args.pop(
                        'filepath_or_buffer')
                    for argname in self.non_spreadsheet_args:
                        self.pickled_args.pop(argname, None)
                return self.pickled_args
        else:
            if self.is_mysql:
                self.sql_validator = MySQLTableValidator(
                    specs=self.specification)
            else:
                self.sql_validator = PostGRETableValidator(
                    specs=self.specification)
            return self.sql_validator.parser_args

    def _non_spreadsheet_args_default(self):
        return ['sep', 'parse_dates', 'nrows', 'names', 'usecols',
                'error_bad_lines', 'dtype', 'header']

    def _set_parser_args(self, specs):
        self.parser_args.update(specs)

    @cached_property
    def _get_error_bad_lines(self):
        return self.specification.get('error_bad_lines', False)

    @cached_property
    def _get_pickle_file(self):
        return self.specification.get('pickle')

    @cached_property
    def _get_pickled_args(self):
        if self.pickle_file is not None:
            with open(self.pickle_file, "r") as fid:
                args = cPickle.load(fid)
            return args
        return {}

    @cached_property
    def _get_exclude_columns(self):
        return self.specification.get("exclude_columns", [])

    @cached_property
    def _get_header(self):
        return self.specification.get("header", 'infer')

    @cached_property
    def _get_column_names(self):
        return self.specification.get("column_names")

    @cached_property
    def _get_converters(self):
        return self.specification.get("converters", None)

    @cached_property
    def _get_md5(self):
        return self.specification.get("md5", "")

    @cached_property
    def _get_na_values(self):
        na_values = self.specification.get("na_values", None)
        if na_values is None:
            na_values = {}
            col_rules = self.specification.get("column_rules", {})
            for colname, rules in col_rules.iteritems():
                if "na_values" in rules:
                    na_values[colname] = rules['na_values']
            if len(na_values) == 0:
                na_values = None
        return na_values

    @cached_property
    def _get_colnames(self):
        usecols = self.specification.get('use_columns')
        if len(self.exclude_columns) > 0:
            if usecols:
                for colname in self.exclude_columns:
                    usecols.remove(colname)
            else:
                usecols = colnames(self.filepath, sep=self.delimiter)
                for colname in self.exclude_columns:
                    usecols.remove(colname)
        else:
            if usecols is None:
                if self.filepath and not self.is_multifile:
                    return colnames(self.filepath, sep=self.delimiter)
        if self.index_col is not None:
            if usecols is not None:
                if self.index_col not in usecols:
                    usecols.append(self.index_col)
        return usecols

    @cached_property
    def _get_nrows(self):
        return self.specification.get('nrows', None)

    @cached_property
    def _get__dtypes(self):
        return self.specification.get('dtypes', {})

    @cached_property
    def _get_delimiter(self):
        return self.specification.get('delimiter', ',')

    # Trait change handlers

    def _specfile_changed(self):
        if self.specification == {}:
            with open(self.specfile, "r") as f:
                self.specification = yaml.load(f,
                                               Loader=Loader).get(
                    self.name, {})

    def _filepath_default(self):
        return self.specification.get("path")

    def __dtypes_items_changed(self):
        self.dtypes = self._dtypes

    # Trait initializers

    def _specification_default(self):
        if op.isfile(self.specfile):
            with open(self.specfile, 'r') as f:
                data = yaml.load(f, Loader=Loader).get(self.name, {})
            return data
        return {}

    def _dtypes_default(self):
        return self._dtypes

    def _df_rules_default(self):
        return {}
