import os
import sys

import audb
import audeer


sys.path.append(os.path.abspath("."))


# Project -----------------------------------------------------------------
project = "audbcards"
author = "Hagen Wierstorf, Christian Geng"
version = audeer.git_repo_version()
title = project


# General -----------------------------------------------------------------
master_doc = "index"
source_suffix = ".rst"
exclude_patterns = ["build", "Thumbs.db", ".DS_Store", "api-src"]
pygments_style = None
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",  # support for Google-style docstrings
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_apipages",
    "audbcards.sphinx",
]
# Disable Gitlab as we need to sign in
linkcheck_ignore = [
    "https://gitlab.audeering.com",
    "https://sphinx-doc.org/",
    ".*/index.html",  # ignore relative links
]
intersphinx_mapping = {
    "audb": ("https://audeering.github.io/audb/", None),
}
# Configure audbcards extension
audbcards_datasets = [
    (
        "data-public",
        "data-public",
        audb.Repository(
            name="data-public",
            host="https://audeering.jfrog.io/artifactory",
            backend="artifactory",
        ),
        True,
    ),
]


# HTML --------------------------------------------------------------------
html_theme = "sphinx_audeering_theme"
html_theme_options = {
    "display_version": True,
    "logo_only": False,
    "footer_links": False,
}
html_context = {
    "display_github": True,
}

html_title = title
