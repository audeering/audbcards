import audbcards
import pytest

from audbcards.core.dataset import create_datacard_page_from_template
from audbcards.core.dataset import create_datasets_page_from_template
from audbcards.core.dataset import create_datacard_page


def test_datacard_from_template_class(db, default_template):
    """Create Datacard using Datacards class."""

    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION)
    dc = audbcards.Datacard(dataset)
    _ = dc._render_template()


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
