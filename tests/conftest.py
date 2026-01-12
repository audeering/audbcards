from collections.abc import Sequence
import json
import os

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


@pytest.fixture(scope="function")
def cache(tmpdir):
    """Local cache folder."""
    return audeer.mkdir(audeer.path(tmpdir, "cache"))


@pytest.fixture(scope="session", autouse=True)
def audb_cache(tmpdir_factory):
    """Local audb cache folder."""
    cache = tmpdir_factory.mktemp("audb-cache")
    current_cache = audb.config.CACHE_ROOT
    current_shared_cache = audb.config.SHARED_CACHE_ROOT
    audb.config.CACHE_ROOT = cache
    audb.config.SHARED_CACHE_ROOT = cache

    yield

    audb.config.CACHE_ROOT = current_cache
    audb.config.SHARED_CACHE_ROOT = current_shared_cache


@pytest.fixture(scope="session")
def repository(tmpdir_factory):
    """Local audb repository only visible inside the tests."""
    name = "data-local"
    host = tmpdir_factory.mktemp("repo")
    audeer.mkdir(audeer.path(host, name))
    current_repositories = audb.config.REPOSITORIES
    repository = audb.Repository(
        name=name,
        host=host,
        backend="file-system",
    )
    audb.config.REPOSITORIES = [repository]

    yield repository

    audb.config.REPOSITORIES = current_repositories


@pytest.fixture(scope="session")
def bare_db(
    tmpdir_factory,
    repository,
    audb_cache,
):
    r"""Publish and load a bare database.

    The name of the database will be ``bare_db``.

    The database has no schemes,
    no table,
    and no media files.

    """
    name = "bare_db"

    db_path = tmpdir_factory.mktemp(name)

    db = audformat.Database(name=name)
    db.save(db_path)

    # Publish and load database
    audb.publish(db_path, pytest.VERSION, repository)
    db = audb.load(name, version=pytest.VERSION, verbose=False)
    return db


@pytest.fixture(scope="session")
def json_db(
    tmpdir_factory,
    repository,
    audb_cache,
):
    r"""Publish and load a database with json files.

    The name of the database will be ``json_db``.

    The database contains two JSON files,
    and a corresponding ``"json"`` table
    without a column.

    """
    name = "json_db"

    db_path = tmpdir_factory.mktemp(name)

    db = audformat.Database(
        name=name,
        source="https://github.com/audeering/audbcards",
        usage="unrestricted",
        expires=None,
        languages=[],
        description="Json database.",
        author="H Wierstorf",
        license=audformat.define.License.CC0_1_0,
    )

    # Table 'json'
    index = audformat.filewise_index(["f0.json", "f1.json"])
    db["json"] = audformat.Table(index)
    path = audeer.path(db_path, "f0.json")
    var = [
        {
            "role": "human",
            "text": "What's the weather?",
        },
        {
            "role": "assistant",
            "text": "Nice",
        },
    ]
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(var, fp, ensure_ascii=False, indent=2)
    path = audeer.path(db_path, "f1.json")
    var = [
        {
            "role": "human",
            "text": "What's the capital of France?",
        },
        {
            "role": "assistant",
            "text": "Paris",
        },
    ]
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(var, fp, ensure_ascii=False, indent=2)

    db.save(db_path)

    # Publish and load database
    audb.publish(db_path, pytest.VERSION, repository)
    db = audb.load(name, version=pytest.VERSION, verbose=False)
    return db


@pytest.fixture(scope="session")
def minimal_db(
    tmpdir_factory,
    repository,
    audb_cache,
):
    r"""Publish and load a minimal database.

    The name of the database will be ``minimal_db``.

    The database has no schemes
    and a single filewise table.

    Further it contains a single file
    with a length of 0.1 s.

    """
    name = "minimal_db"

    db_path = tmpdir_factory.mktemp(name)

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
    return db


@pytest.fixture(scope="session")
def medium_db(
    tmpdir_factory,
    repository,
    audb_cache,
):
    r"""Publish and load a medium test database.

    The name of the database will be ``medium_db``.

    The database contains
    several schemes,
    filewise, segmented, and misc tables,
    and audio files that are suited as an example.

    """
    name = "medium_db"

    db_path = tmpdir_factory.mktemp(name)

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
    return db


@pytest.fixture(scope="session")
def mixed_db(
    tmpdir_factory,
    repository,
    audb_cache,
):
    r"""Publish and load a mixed database.

    The name of the database will be ``mixed_db``.

    The database contains one WAV and one JSON file,
    and corresponding ``"audio"`` and ``"json"`` tables.

    """
    name = "mixed_db"

    db_path = tmpdir_factory.mktemp(name)

    db = audformat.Database(
        name=name,
        source="https://github.com/audeering/audbcards",
        usage="unrestricted",
        expires=None,
        languages=[],
        description="Mixed database.",
        author="H Wierstorf, C Geng, B E Abrougui",
        license=audformat.define.License.CC0_1_0,
    )

    # Table 'audio'
    db.schemes["transcription"] = audformat.Scheme("str")
    index = audformat.filewise_index(["f0.wav", "f1.wav"])
    db["audio"] = audformat.Table(index)
    db["audio"]["transcription"] = audformat.Column()
    db["audio"]["transcription"].set(["Hello World", ""])
    sampling_rate = 8000
    path = audeer.path(db_path, "f0.wav")
    signal = np.random.normal(0, 0.1, (1, int(0.1 * sampling_rate)))  # 0.1 s
    audiofile.write(path, signal, sampling_rate, normalize=True)
    path = audeer.path(db_path, "f1.wav")
    signal = np.random.normal(0, 0.1, (1, 0))  # 0.0 s
    audiofile.write(path, signal, sampling_rate, normalize=False)

    # Table 'json'
    db.schemes["turns"] = audformat.Scheme("int")
    index = audformat.filewise_index(["c0.json"])
    db["json"] = audformat.Table(index)
    db["json"]["turns"] = audformat.Column()
    db["json"]["turns"].set([2])
    path = audeer.path(db_path, "c0.json")
    var = [
        {
            "role": "human",
            "audio": "f0.wav",
            "transcription": "Hello World",
        },
        {
            "role": "assistant",
            "audio": "f1.wav",
        },
    ]
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(var, fp, ensure_ascii=False, indent=2)

    db.save(db_path)

    # Publish and load database
    audb.publish(db_path, pytest.VERSION, repository)
    db = audb.load(name, version=pytest.VERSION, verbose=False)
    return db


def create_audio_files(
    db: audformat.Database,
    db_path: str,
    durations: Sequence[float],
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
