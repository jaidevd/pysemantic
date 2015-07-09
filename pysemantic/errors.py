#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@schrodinger.local>
#
# Distributed under terms of the BSD 3-clause license.

"""Errors."""


class MissingProject(Exception):

    """Error raised when project is not found."""


class MissingConfigError(Exception):

    """Error raised when the pysemantic configuration file is not found."""


class ParserArgumentError(Exception):

    """Error raised when no valid parser arguments are inferred from the
    schema."""
