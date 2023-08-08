import os
from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="wavesep",
    packages=find_packages(include=["wavesep", "wavesep.*"]),
    version="0.1.0",
    description="",
    author="Zhenghan Fang",
    author_email="zfang23@jhu.edu",
    long_description=read("README.md"),
)
