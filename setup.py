# setup.py
import os

from setuptools import setup, find_packages
os.system('mv fams_dnn ptagnn')
setup(
    name="ptagnn",
    version='1.0',
    packages=find_packages(),
    author='wqzhang-group',
    description='MLIP',
)