from setuptools import setup, find_packages

setup(
    name='SpectroscoPy',
    version='0.1.0',
    packages=find_packages(include=['data_core', 'data_core.*'])
)
