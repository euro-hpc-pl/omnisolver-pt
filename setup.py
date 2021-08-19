"""Setup script for omnisolver-pt project."""
from setuptools import find_namespace_packages, setup

with open("README.md") as readme:
    long_description = readme.read()


setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="omnisolver-pt",
    entry_points={
        "omnisolver": ["pt = omnisolver.pt"],
    },
    setup_requires=["setuptools-scm>=4.1.2"],
    install_requires=[
        "dimod>=0.9.10",
        "numpy~=1.19.4",
        "pluggy~=0.13.1",
        "PyYAML~=5.3.1",
        "pandas~=1.1.4",
        "omnisolver",
    ],
    tests_require=["pytest==6.1.2", "pytest-mock==3.3.1"],
    packages=find_namespace_packages(exclude=["tests"]),
    package_data={"omnisolver.pt": ["pt.yml"]},
)
