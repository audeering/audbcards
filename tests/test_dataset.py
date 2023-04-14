import audb
import audbcards
import jinja2
import pytest

from audbcards.core.dataset import create_datacard_page_from_template
from audbcards.core.dataset import create_datacard_page
from audbcards.core.dataset import Datacard


def test_datacard_from_template_class(db, default_template):

    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION)
    dc = Datacard(dataset)
    content = dc._render_template()


def test_datacard_from_template(db, default_template):

    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION)

    create_datacard_page_from_template(dataset)


@pytest.mark.skip(reason="currently writing to wrong directory")
def test_datacard_original(db, default_template):

    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION)

    create_datacard_page(dataset)
