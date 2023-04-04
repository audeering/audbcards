import pytest

import audb


def test_db_fixture(db):

    assert db.name == pytest.NAME

    db_loaded = audb.load(
        pytest.NAME,
        version=pytest.VERSION,
        verbose=False,
    )
    assert db_loaded == db
    assert audb.repository(pytest.NAME, pytest.VERSION) == pytest.REPOSITORY


def test_publish_db_fixture(publish_db):

    db_loaded = audb.load(
        pytest.NAME,
        version=pytest.VERSION,
        verbose=False,
    )
    assert db_loaded.name == pytest.NAME
    assert audb.repository(pytest.NAME, pytest.VERSION) == pytest.REPOSITORY
