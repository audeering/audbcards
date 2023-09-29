import re

import pytest

import audbcards
from audbcards.core.dataset import create_datasets_page


def test_datacard_lines_similar(db, default_template, cache):
    """Create datacard using jinja2 via Dataset and Datacard.

    The assertions for exact identity are currently too strict.
    Therefore this uses text similarities as obtained from
    the difflib builtins. These are based on

    - average (or rather median and mean) similarities per line
    - percentage of lines differing between original and rendered

    """

    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION, cache)
    dc = audbcards.Datacard(dataset)
    content = dc._render_template()
    content = content.rstrip()

    # Remove lines that depent on author/local machine
    for pattern in [
            re.compile('^published.*$', flags=(re.MULTILINE)),
            re.compile('^repository.*$', flags=(re.MULTILINE)),
    ]:
        content = re.sub(pattern, '', content)
        default_template = re.sub(pattern, '', default_template)

    assert content == default_template


def test_create_datasets_page(db):

    datasets = [audbcards.Dataset(pytest.NAME, pytest.VERSION)] * 4
    create_datasets_page(datasets, ofbase="datasets_page")
