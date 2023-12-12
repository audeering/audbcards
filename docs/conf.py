import os
import sys

import audb
import audeer


sys.path.append(os.path.abspath('.'))


# Project -----------------------------------------------------------------
project = 'audbcards'
author = 'Hagen Wierstorf, Christian Geng'
version = audeer.git_repo_version()
title = project


# General -----------------------------------------------------------------
master_doc = 'index'
source_suffix = '.rst'
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']
pygments_style = None
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'audbcards.sphinx',
]
# Disable Gitlab as we need to sign in
linkcheck_ignore = [
    'https://gitlab.audeering.com',
    'https://sphinx-doc.org/',
    '.*/index.html',  # ignore relative links
]
intersphinx_mapping = {
    'audb': ('https://audeering.github.io/audb/', None),
}
# Configure audbcards extension
audbcards_datasets = [
    (
        'data-public',
        'data-public',
        audb.Repository(
            name='data-public',
            host='https://audeering.jfrog.io/artifactory',
            backend='artifactory',
        ),
        True,
    ),
    # (
    #     'data-private',
    #     'data-private',
    #     audb.Repository(
    #         name='data-private',
    #         host='https://audeering.jfrog.io/artifactory',
    #         backend='artifactory',
    #     ),
    #     True,
    # ),
    # (
    #     'all-repos',
    #     'All repos',
    #     [
    #         audb.Repository(
    #             name='data-public',
    #             host='https://audeering.jfrog.io/artifactory',
    #             backend='artifactory',
    #         ),
    #         audb.Repository(
    #             name='data-private',
    #             host='https://audeering.jfrog.io/artifactory',
    #             backend='artifactory',
    #         ),
    #     ],
    #     False,
    # ),
]


# HTML --------------------------------------------------------------------
html_theme = 'sphinx_audeering_theme'
html_theme_options = {
    'display_version': True,
    'logo_only': False,
}
html_title = title
