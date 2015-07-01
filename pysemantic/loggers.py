#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""Loggers"""

import os
import os.path as op
import logging
import time


LOGDIR = op.join(op.expanduser("~"), ".pysemantic")
if not op.exists(LOGDIR):
    os.mkdir(LOGDIR)


def setup_logging(project_name):
    logfile = "{0}_{1}.log".format(project_name, time.time())
    logging.basicConfig(filename=op.join(LOGDIR, logfile),
                        level=logging.INFO)
    logging.info("Project {0} started.".format(project_name))
