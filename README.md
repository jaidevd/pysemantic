[![Build Status](https://travis-ci.org/motherbox/pysemantic.svg?branch=master)](https://travis-ci.org/motherbox/pysemantic)
[![Coverage Status](https://coveralls.io/repos/motherbox/pysemantic/badge.svg?branch=master)](https://coveralls.io/r/motherbox/pysemantic?branch=master)
[![Code Health](https://landscape.io/github/motherbox/pysemantic/master/landscape.svg?style=plastic)](https://landscape.io/github/motherbox/pysemantic/master)

# pysemantic
A traits based data validation module for pandas data structures.

What it is
==========
* A data validator which does its validation based on a centralized data
dictionary.
* Data type validator
* Range and constraint validator
* Post validation actions:
  - enforcement of data types
  - warnings and errors based on 'acceptability' of data
  - verification of converted data
* Data dictionary parser
* Container for rules where rules are native Python objects.


What it is not
==============
* data parserr
* no unnecessary typecasting
* no file I/O
* no "analysis" on the data other than what is required for verification
* no automatic inference of data types
