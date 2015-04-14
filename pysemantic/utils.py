#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Misecellaneous bells and whistles
"""

import json


class TypeEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, type):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)


def colnames(filename, **kwargs):
    """
    Read the column names of a delimited file, without actually reading the
    whole file. This is simply a wrapper around `pandas.read_csv`, which reads
    only one row and returns the column names.

    Parameters:
    -----------

    filename: Path to the file to be read

    kwargs: Arguments to be passed to the `pandas.read_csv`

    """

    if 'nrows' in kwargs:
        UserWarning("The nrows parameter is pointless here. This function only"
                    "reads one row.")
        kwargs.pop('nrows')
    import pandas as pd
    return pd.read_csv(filename, nrows=1, **kwargs).columns.tolist()
