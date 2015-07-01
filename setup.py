from setuptools import setup, find_packages
import os.path as op

CONF_PATH = op.join(op.expanduser("~"), "pysemantic.conf")
if not op.exists(CONF_PATH):
    with open(CONF_PATH, "w") as fid:
        fid.write("# Config file added by the pysemantic setup script.")
        fid.write("\n")
    print "Config file added at {}".format(CONF_PATH)

NAME = "pysemantic"

setup(
    name=NAME,
    version='0.1.1',
    author='Jaidev Deshpande',
    author_email='deshpande.jaidev@gmail.com',
    description="A traits based data validation module for pandas data structures.",
    url="https://github.com/motherbox/pysemantic",
    long_description=open("README.rst").read(),
    entry_points={
        'console_scripts': ['semantic = pysemantic.cli:main'],
               },
    packages=find_packages(),
    install_requires=['pyyaml', 'traits', 'pandas', 'docopt']
)
