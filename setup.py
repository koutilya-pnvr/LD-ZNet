from setuptools import setup, find_packages

setup(
    name='ld-znet',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)