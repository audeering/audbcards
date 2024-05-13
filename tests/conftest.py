import os
import typing

import numpy as np
import pandas as pd
import pytest

import audb
import audeer
import audformat
import audiofile


pytest.VERSION = "1.0.0"
pytest.ROOT = os.path.dirname(os.path.realpath(__file__))
pytest.TEMPLATE_DIR = audeer.mkdir(
    os.path.join(
        pytest.ROOT,
        "test_data",
        "rendered_templates",
    )
)


@pytest.fixture
def cache(tmpdir, scope="function"):
    """Local cache folder."""
    return audeer.mkdir(audeer.path(tmpdir, "cache"))


@pytest.fixture
def audb_cache(tmpdir, scope="session", autouse=True):
    """Local audb cache folder."""
    cache = audeer.mkdir(audeer.path(tmpdir, "audb-cache"))
    audb.config.CACHE_ROOT = cache
    audb.config.SHARED_CACHE_ROOT = cache


@pytest.fixture
def repository(tmpdir, scope="session"):
    """Local audb repository only visible inside the tests."""
    name = "data-local"
    host = audeer.mkdir(audeer.path(tmpdir, "repo"))
    audeer.mkdir(audeer.path(host, name))
    repository = audb.Repository(
        name=name,
        host=host,
        backend="file-system",
    )
    audb.config.REPOSITORIES = [repository]
    return repository


@pytest.fixture
def bare_db(
    tmpdir,
    repository,
    audb_cache,
    scope="session",
    autouse=True,
):
    r"""Publish and load a bare database.

    The name of the database will be ``bare_db``.

    The database has no schemes,
    no table,
    and no media files.

    """
    name = "bare_db"

    db_path = audeer.mkdir(audeer.path(tmpdir, name))

    db = audformat.Database(name=name)
    db.save(db_path)

    # Publish and load database
    audb.publish(db_path, pytest.VERSION, repository)
    db = audb.load(name, version=pytest.VERSION, verbose=False)
    tmp_root = str(tmpdir.parts()[1])
    assert db.root.startswith(tmp_root)
    return db


@pytest.fixture
def minimal_db(
    tmpdir,
    repository,
    audb_cache,
    scope="session",
    autouse=True,
):
    r"""Publish and load a minimal database.

    The name of the database will be ``minimal_db``.

    The database has no schemes
    and a single filewise table.

    Further it contains a single file
    with a length of 0.1 s.

    """
    name = "minimal_db"

    db_path = audeer.mkdir(audeer.path(tmpdir, name))

    db = audformat.Database(
        name=name,
        source="https://github.com/audeering/audbcards",
        usage="unrestricted",
        expires=None,
        languages=[],
        description="Minimal database.",
        author="H Wierstorf, C Geng, B E Abrougui",
        license=audformat.define.License.CC0_1_0,
    )

    # Table 'files'
    index = audformat.filewise_index(["f0.wav"])
    db["files"] = audformat.Table(index)
    db["files"]["speaker"] = audformat.Column()
    db["files"]["speaker"].set([0])

    # Create audio files and store database
    durations = [0.1]  # s
    create_audio_files(db, db_path, durations)
    db.save(db_path)

    # Publish and load database
    audb.publish(db_path, pytest.VERSION, repository)
    db = audb.load(name, version=pytest.VERSION, verbose=False)
    tmp_root = str(tmpdir.parts()[1])
    assert db.root.startswith(tmp_root)
    return db


@pytest.fixture
def medium_db(
    tmpdir,
    repository,
    audb_cache,
    scope="session",
    autouse=True,
):
    r"""Publish and load a medium test database.

    The name of the database will be ``medium_db``.

    The database contains
    several schemes,
    filewise, segmented, and misc tables,
    and audio files that are suited as an example.

    """
    name = "medium_db"

    db_path = audeer.mkdir(audeer.path(tmpdir, name))

    db = audformat.Database(
        name=name,
        source="https://github.com/audeering/audbcards",
        usage="unrestricted",
        expires=None,
        languages=["eng", "de"],
        description="Medium database. | Some description |.",
        author="H Wierstorf, C Geng, B E Abrougui",
        organization="audEERING",
        license=audformat.define.License.CC0_1_0,
    )

    # Misc table 'speaker'
    db.schemes["age"] = audformat.Scheme(
        "int",
        minimum=0,
        description="Age of speaker",
    )
    db.schemes["gender"] = audformat.Scheme(
        "str",
        labels=["female", "male"],
        description="Gender of speaker",
    )
    index = pd.Index(
        [0, 1],
        dtype="Int64",
        name="speaker",
    )
    db["speaker"] = audformat.MiscTable(index)
    db["speaker"]["age"] = audformat.Column(scheme_id="age")
    db["speaker"]["age"].set([23, 49])
    db["speaker"]["gender"] = audformat.Column(scheme_id="gender")
    db["speaker"]["gender"].set(["female", "male"])

    # Table 'files'
    db.schemes["speaker"] = audformat.Scheme(
        "int",
        labels="speaker",
        description="Speaker IDs.",
    )
    index = audformat.filewise_index(["data/f0.wav", "data/f1.wav"])
    db["files"] = audformat.Table(index)
    db["files"]["speaker"] = audformat.Column(scheme_id="speaker")
    db["files"]["speaker"].set([0, 1])

    # Table 'segments'
    db.schemes["emotion"] = audformat.Scheme(
        "str",
        labels=["angry", "happy", "neutral"],
        description="Emotional class.",
    )
    index = audformat.segmented_index(
        files=["data/f0.wav", "data/f0.wav", "data/f1.wav", "data/f1.wav"],
        starts=[0, 0.5, 0, 150],
        ends=[0.5, 1, 150, 301],
    )
    db["segments"] = audformat.Table(index)
    db["segments"]["emotion"] = audformat.Column(scheme_id="emotion")
    db["segments"]["emotion"].set(["neutral", "neutral", "happy", "angry"])

    # Create audio files and store database
    durations = [1, 301]
    create_audio_files(db, db_path, durations)
    db.save(db_path)

    # Publish and load database
    audb.publish(db_path, pytest.VERSION, repository)
    db = audb.load(name, version=pytest.VERSION, verbose=False)
    tmp_root = str(tmpdir.parts()[1])
    assert db.root.startswith(tmp_root)
    return db


def create_audio_files(
    db: audformat.Database,
    db_path: str,
    durations: typing.Sequence[float],
    *,
    sampling_rate: int = 8000,
    seed: int = 1,
):
    r"""Create audio files with given durations."""
    np.random.seed(seed)
    for n, file in enumerate(list(db["files"].index)):
        path = audeer.path(db_path, file)
        audeer.mkdir(os.path.dirname(path))
        samples = int(durations[n] * sampling_rate)
        signal = np.random.normal(0, 0.1, (1, samples))
        audiofile.write(path, signal, sampling_rate, normalize=True)
