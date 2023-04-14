from setuptools import setup, find_packages

setup(
    name = 'lrcc',
    version = '0.1.0',
    description = "for pytorch model, convert conv op to low rank conv op",
    packages=find_packages(),
    python_requires='>=3.6',
)