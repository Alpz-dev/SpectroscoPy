from setuptools import setup, find_packages

setup(
    name='SpectroscoPy',
    version='0.1.2',
    author='Shreyas Srinivasan',
    packages=find_packages(include=['SpectroscoPy', 'SpectroscoPy.*']),
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      'tqdm',
                      'scalene']
)
