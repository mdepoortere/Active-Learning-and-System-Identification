#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt")) as f:
    requirements = f.read().split()


setup(
    name="nnsetup",
    version="0.0.0",
    description="Tools for active learning and MC-dropout ",
    author="Matthias Depoortere",
    author_email="Depoortere.matthias@gmail.com",
    url="https://github.com/mdepoortere/BALR",
    packages=find_packages(exclude=[]),
    install_requires=requirements,
)
