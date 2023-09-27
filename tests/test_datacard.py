import audbcards
import pytest
import difflib
import numpy as np

from audbcards.core.dataset import create_datasets_page


def test_datacard_from_template_lines_similar(db, default_template):
    """Create datacard using jinja2 via Dataset and Datacard.

    The assertions for exact identity are currently too strict.
    Therefore this uses text similarities as obtained from
    the difflib builtins. These are based on

    - average (or rather median and mean) similarities per line
    - percentage of lines differing between original and rendered

    """

    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION)
    dc = audbcards.Datacard(dataset)
    content = dc._render_template()

    ratios = []
    for line1, line2 in zip(default_template.split("\n"), content.split("\n")):
        sm = difflib.SequenceMatcher(lambda x: x == " ", line1, line2)
        ratios.append(sm.ratio())

    # there are lines that will differ (author, date of dataset)
    # almost all lines are identical, so mean and median should be high.
    assert np.median(np.array(ratios)) == 1
    assert np.mean(np.array(ratios)) > 0.95

    # use differ to check how many lines are identical
    differ = difflib.Differ()
    obtained, expected = content.splitlines(), default_template.splitlines()
    result = list(differ.compare(obtained, expected))
    # more than 90% of the lines should be identical
    ratio_diff = len([x for x in result if x.startswith("-")])
    assert 1. - ratio_diff / len(result) > 0.85


def test_create_datasets_page(db):

    datasets = [audbcards.Dataset(pytest.NAME, pytest.VERSION)] * 4
    create_datasets_page(datasets, ofbase="datasets_page")
