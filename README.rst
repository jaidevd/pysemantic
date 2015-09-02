.. -*- mode: rst -*-

|Travis|_ |Coveralls|_ |Landscape|_ |RTFD|_

.. |Travis| image:: https://travis-ci.org/dataculture/pysemantic.svg?branch=master
.. _Travis: https://travis-ci.org/dataculture/pysemantic

.. |Coveralls| image:: https://coveralls.io/repos/motherbox/pysemantic/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/r/motherbox/pysemantic?branch=master

.. |Landscape| image:: https://landscape.io/github/dataculture/pysemantic/master/landscape.svg?style=flat
.. _Landscape: https://landscape.io/github/dataculture/pysemantic/master

.. |RTFD| image:: https://readthedocs.org/projects/pysemantic/badge/?version=latest
.. _RTFD: https://readthedocs.org/projects/pysemantic/?badge=latest

.. image:: docs/_static/logo.png

pysemantic
==========
A traits based data validation and data cleaning module for pandas data structures.

Dependencies
------------
* Traits
* PyYaml
* pandas
* docopt

Quick Start
-----------

Installing with pip
+++++++++++++++++++

Run::

    $ pip install pysemantic

Installing from source
++++++++++++++++++++++

You can install pysemantic by cloning this repository, installing the
dependencies and running::

    $ python setup.py install

in the root directory of your local clone.

Usage
+++++

Create an empty file named ``pysemantic.conf`` in your home directory. This can be as simple as running::

$ touch ~/pysemantic.conf

After installing pysemantic, you should have a command line script called
``semantic``. Try it out by running::

$ semantic list

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
using the ``add`` subcommand of the ``semantic`` script as follows::

$ semantic add pysemantic_demo

As you can see, this does not fit the supported usage of the ``add`` subcommand.
We additionally need a file containing the specifications for this project.
(Note that this file, containing the specifications, is referred to throughout
the documentation interchangeably as a *specfile* or a *data dictionary*.)
Before we create this file, let's download the well known Fisher iris datset,
which we will use as the sample dataset for this demo. You can download it
`here <https://raw.githubusercontent.com/motherbox/pysemantic/master/pysemantic/tests/testdata/iris.csv>`_.

Once the dataset is downloaded, fire up your favourite text editor and create a
file named ``demo_specs.yaml``. Fill it up with the following content.

.. code-block:: yaml

    iris:
      path: /absolute/path/to/iris.csv

Now we can use this file as the data dictionary of the ``pysemantic_demo``
project. Let's tell pysemantic that we want to do so, by running the following
command::

$ semantic add pysemantic_demo /path/to/demo_specs.yaml

We're all set. To see how we did, start a Python interpreter and type the
following statements::

>>> from pysemantic import Project
>>> demo = Project("pysemantic_demo")
>>> iris = demo.load_dataset("iris")

Voila! The Python object named ``iris`` is actually a pandas DataFrame containing
the iris dataset! Well, nothing really remarkable so far. In fact, we cloned
and installed a module, wrote two seemingly unnecessary files, and typed three
lines of Python code to do something that could have been achieved by simply
writing::

>>> iris = pandas.read_csv("/path/to/iris.csv")

Most datasets, however, are not as well behaved as this one. In fact they can
be a nightmare to deal with. Pysemantic can be far more intricate and far
smarter than this when dealing with mangled, badly encoded, ugly data with
inconsistent data types. Check the IPython notebooks in the examples to see how to use Pysemantic for
such data.
