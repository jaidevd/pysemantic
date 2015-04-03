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
    version='0.0.1',
    author='Jaidev Deshpande',
    author_email='jaidev@dataculture.in',
    entry_points={
        'console_scripts': ['semantic = pysemantic.cli:main'],
               },
    packages=find_packages(),
)
