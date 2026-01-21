from setuptools import setup, find_packages
import os

# Read the content of your README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="femanalyzer",
    version="0.1.0",  # Sourced from femanalyzer/__init__.py
    author="msunkaradtt",
    description="A Python package for analyzing the effect of Finite Element (FEA) simulations on 3D surface roughness.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/msunkaradtt/femanalyzer",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "cascadio>=0.0.17",
        "numpy>=2.3.4",
        "scikit-fem>=11.0.0",
        "pygmsh>=7.1.17",
        "trimesh>=4.9.0",
        "scikit-learn>=1.7.2",
        "pyvista>=0.46.4",
        "matplotlib>=3.10.7",
        "gmsh>=4.15.0",
        "meshio>=5.3.5",
        "scipy>=1.16.3",
        "networkx>=3.5",
        "requests>=2.32.5",
        "rich>=14.2.0",
        "vtk>=9.5.2",
    ],
)