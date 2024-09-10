"""Configuration file for the Sphinx documentation builder."""


import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "PyProBE"
copyright = (
    "2024, Thomas Holland, Electrochemical Science and Engineering Group, "
    "Imperial College London"
)

author = "Thomas Holland"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_tabs.tabs",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "nbsphinx",
    "sphinxcontrib.autodoc_pydantic",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_theme_options = {"collapse_navigation": True, "show_nav_level": 4}


# -- Options for autodoc extension -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_typehints = "description"
autodoc_default_options = {
    "exclude-members": "model_post_init, Config",
    "show-inheritance": True,
}

# -- sphinxcontrib-bibtex configuration --------------------------------------
bibtex_bibfiles = ["../../CITATIONS.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = """.. rubric:: References"""
bibtex_reference_style = "author_year"
bibtex_tooltips = True

# -- nbsphinx configuration --------------------------------------------------
autosummary_generate = True
nbsphinx_execute = "always"  # Always execute notebooks
nbsphinx_allow_errors = True  # Raise exceptions when notebooks raise errors

# -- sphinxcontrib-autodoc_pydantic configuration ----------------------------
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
