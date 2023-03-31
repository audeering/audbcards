import pytest

import audb


def test_db_fixture(db):

    assert db.name == pytest.NAME

    db_loaded = audb.load(
        pytest.NAME,
        version=pytest.VERSION,
        verbose=False,
        full_path=False,
    )
    # Remove field added by audb
    del db_loaded.meta['audb']
    assert db_loaded == db
