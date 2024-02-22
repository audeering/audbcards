import pytest

import audb


@pytest.mark.parametrize(
    "db",
    [
        "minimal_db",
        "medium_db",
    ],
)
def test_db_fixture(repository, db, request):
    r"""Check if fixtures for publishing databases work as expected."""
    db = request.getfixturevalue(db)
    db_loaded = audb.load(
        db.name,
        version=pytest.VERSION,
        verbose=False,
    )
    assert db_loaded == db
    assert audb.repository(db.name, pytest.VERSION) == repository
