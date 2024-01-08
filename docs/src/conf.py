"""Configuration file for the Sphinx documentation builder."""
import os
import sys


sys.path.insert(0, os.path.abspath(".."))

project = "MS-Zarr Converter"
copyright = "2024, SKA Observatory"
author = "Max Maunder, Sean Stansill"
extensions = ["sphinx_rtd_theme", "myst_parser", "sphinx_gitstamp"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
html_theme = "ska_ser_sphinx_theme"
