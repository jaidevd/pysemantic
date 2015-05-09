[![Build Status](https://travis-ci.org/motherbox/pysemantic.svg?branch=master)](https://travis-ci.org/motherbox/pysemantic)
[![Coverage Status](https://coveralls.io/repos/motherbox/pysemantic/badge.svg?branch=master)](https://coveralls.io/r/motherbox/pysemantic?branch=master)
[![Code Health](https://landscape.io/github/motherbox/pysemantic/master/landscape.svg?style=plastic)](https://landscape.io/github/motherbox/pysemantic/master)
[![Documentation Status](https://readthedocs.org/projects/pysemantic/badge/?version=latest)](https://readthedocs.org/projects/pysemantic/?badge=latest)

[logo][docs/_static/logo.png]

# pysemantic
A traits based data validation and data cleaning module for pandas data structures.

Dependencies
------------
* Traits
* PyYaml
* pandas
* docopt

Quick Start
===========

You can install pysemantic by cloning this repository, installing the
dependencies and running

`$ python setup.py develop`

in the root directory of your local clone. Next, create an empty file named
`pysemantic.conf` in your home directory. This can be as simple as running

`$ touch ~/pysemantic.conf`

After installing pysemantic, you should have a command line script called
`semantic`. Try it out by running

`$ semantic list`

This should do nothing. This means that you don't have any projects regiestered
under pysemantic. A _project_ in pysemantic is just a collection of _datasets_.
pysemantic manages your datasets like an IDE manages source code files in that
it groups them under different projects, and each project has it's own tree
structure, build toolchains, requirements, etc. Similarly, different
pysemantic projects group under them a set of datasets, and manages them
depending on their respective user-defined specifications. Projects are
uniquely identified by their names.

For now, let's add and configure a demo project called, simply,
"pysemantic_demo". You can create a project and register it with pysemantic
using the `add` subcommand of the `semantic` script as follows:

```bash
$ semantic add pysemantic_demo
```

As you can see, this does not fit the supported usage of the `add` subcommand.
We additionally need a file containing the specifications for this project.
(Note that this file, containing the specifications, is referred to throughout
the documentation interchangeably as a _specfile_ or a _data dictionary_.)
Before we create this file, let's download the well known Fisher iris datset,
which we will use as the sample dataset for this demo. You can download it
[here](https://raw.githubusercontent.com/motherbox/pysemantic/master/pysemantic/tests/testdata/iris.csv).

Once the dataset is downloaded, fire up your favourite text editor and create a
file named `demo_specs.yaml`. Fill it up with the following content.

```yaml
iris:
  path: /absolute/path/to/iris.csv
```

Now we can use this file as the data dictionary of the `pysemantic_demo`
project. Let's tell pysemantic that we want to do so, by running the following
command.

```bash
$ semantic add pysemantic_demo /path/to/demo_specs.yaml
```

We're all set. To see how we did, start a Python interpreter and type the
following statements.

```python
>>> from pysemantic import Project
>>> demo = Project("pysemantic_demo")
>>> iris = demo.load_dataset("iris")
```

Voila! The Python object named `iris` is actually a pandas DataFrame containing
the iris dataset! Well, nothing really remarkable so far. In fact, we cloned
and installed a module, wrote two seemingly unnecessary files, and typed three
lines of Python code to do something that could have been achieved by simply
writing:

```python
>>> iris = pandas.read_csv("/path/to/iris.csv")
```

Most datasets, however, are not as well behaved as this one. In fact they can
be a nightmare to deal with. Pysemantic can be far more intricate and far
smarter than this when dealing with mangled, badly encoded, ugly data with
inconsistent data types. Check the IPython notebooks in the examples to see how to use Pysemantic for
such data.

What it is
==========
* A data validator which does its validation based on the specifications written in a centralized data dictionary.
* Data type validator
* Range and constraint validator
* Data dictionary parser
* Container for rules where rules are native Python objects.


What it is not
==============
* data parser
* no unnecessary typecasting
* no file I/O
* no "analysis" on the data other than what is required for verification
