import audbcards
import pytest
import difflib
import numpy as np

from audbcards.core.dataset import create_datacard_page_from_template
from audbcards.core.dataset import create_datasets_page_from_template
from audbcards.core.dataset import create_datacard_page


def test_datacard_from_template_lines_similar(db, default_template):
    """Create datacard using jinja2 via Dataset and Datacard."""

    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION)
    dc = audbcards.Datacard(dataset)
    content = dc._render_template()

    ratios = []
    for line1, line2 in zip(default_template.split('\n'), content.split('\n')):
        sm = difflib.SequenceMatcher(lambda x: x == " ", line1, line2)
        ratios.append(sm.ratio())

    # there are lines that will differ (author, date of dataset)
    # almost all lines are identical, so mean and median should be high.
    assert np.median(np.array(ratios)) == 1
    assert np.mean(np.array(ratios)) > .95

    # use differ to check how many lines are identical
    differ = difflib.Differ()
    obtained, expected = content.splitlines(), default_template.splitlines()
    result = list(differ.compare(obtained, expected))
    # more than 90% of the lines should be identical
    ratio_diff = len([x for x in result if x.startswith('-')])
    assert 1 - ratio_diff / len(result) > .9


@pytest.mark.skip(reason='currently writing to wrong directory')
def test_datacard_from_template(db, default_template):
    """Create Datacard using wrapper function."""

    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION)

    create_datacard_page_from_template(dataset)


@pytest.mark.skip(reason='currently writing to wrong directory')
def test_datacard_original(db, default_template):

    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION)

    create_datacard_page(dataset)


def test_create_datasets_page(db):

    datasets = [audbcards.Dataset(pytest.NAME, pytest.VERSION)] * 4
    create_datasets_page_from_template(datasets,
                                       of_basename='datasets_from_template')
