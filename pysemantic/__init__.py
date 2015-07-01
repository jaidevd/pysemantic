#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@schrodinger.local>
#
# Distributed under terms of the BSD 3-clause license.


from pysemantic.project import Project

__version__ = "0.1.1"


def test():
    """Interactive loader for tests."""
    import unittest
    unittest.main(module='pysemantic.tests', exit=False)

__all__ = ['Project', 'test']
