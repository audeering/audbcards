import os
import posixpath

import numpy as np
import pandas as pd
import pytest

import audb
import audeer
import audformat
import audiofile

import audbcards


@pytest.mark.parametrize(
    "db",
    [
        "medium_db",
    ],
)
def test_dataset_property_scope(tmpdir, db, request):
    r"""Test visibility of properties in local and global scopes."""
    db = request.getfixturevalue(db)

    dataset_cache = audeer.mkdir(tmpdir, "cache")
    dataset = audbcards.Dataset(
        db.name,
        pytest.VERSION,
        cache_root=dataset_cache,
    )

    props = [x for x in dataset._cached_properties().keys()]

    # should not exist in local scope
    for prop in props:
        assert prop not in vars()

    # should not exist in global scope either
    for prop in props:
        assert prop not in globals()

    # ensure Dataset has desired attributes
    for prop in props:
        assert hasattr(dataset, prop)

    # dummy identifier must be in local scope
    repository = "foo"  # noqa F841
    assert "repository" in vars()


@pytest.mark.parametrize(
    "db",
    [
        "medium_db",
    ],
)
def test_dataset(audb_cache, tmpdir, repository, db, request):
    r"""Test audbcards.Dataset object and all its properties."""
    db = request.getfixturevalue(db)

    dataset_cache = audeer.mkdir(tmpdir, "cache")
    dataset = audbcards.Dataset(
        db.name,
        pytest.VERSION,
        cache_root=dataset_cache,
    )
    backend_interface = repository.create_backend_interface()

    # __init__
    assert dataset.name == db.name
    assert dataset.version == pytest.VERSION
    assert dataset.repository_object == repository
    with backend_interface.backend:
        # Compare only string,
        # as backends are not identical
        assert str(dataset.backend) == str(backend_interface)
    expected_header = audb.info.header(
        db.name,
        version=pytest.VERSION,
        cache_root=audb_cache,
    )
    assert str(dataset.header) == str(expected_header)
    expected_deps = audb.dependencies(
        db.name,
        version=pytest.VERSION,
        cache_root=audb_cache,
    )
    expected_df = expected_deps()
    pd.testing.assert_frame_equal(dataset.deps(), expected_df)

    # archives
    expected_archives = len(expected_df.loc[expected_deps.media].archive.unique())
    assert dataset.archives == expected_archives

    # bit depths
    expected_bit_depths = sorted(
        list(
            set(
                [
                    audiofile.bit_depth(file)
                    for file in db.files
                    if audiofile.bit_depth(file)
                ]
            )
        )
    )
    assert dataset.bit_depths == expected_bit_depths

    # channels
    expected_channels = sorted(
        list(set([audiofile.channels(file) for file in db.files]))
    )
    assert dataset.channels == expected_channels

    # duration
    expected_duration = db.files_duration(db.files).sum()
    assert dataset.duration == expected_duration

    # files
    expected_files = len(db.files)
    assert dataset.files == expected_files

    # file_durations
    expected_file_durations = [
        expected_deps.duration(file) for file in expected_deps.media
    ]
    assert dataset.file_durations == expected_file_durations

    # formats
    expected_formats = sorted(
        list(set([audeer.file_extension(file) for file in db.files]))
    )
    assert dataset.formats == expected_formats

    # license
    expected_license = db.license or "Unknown"
    assert dataset.license == expected_license

    # license link
    if db.license_url is None or len(db.license_url) == 0:
        expected_license_link = None
    else:
        expected_license_link = db.license_url
    assert dataset.license_link == expected_license_link

    with backend_interface.backend:
        # publication_date:
        expected_publication_date = backend_interface.date(
            backend_interface.join("/", db.name, "db.yaml"),
            pytest.VERSION,
        )
        assert dataset.publication_date == expected_publication_date

        # publication_owner
        expected_publication_owner = backend_interface.owner(
            backend_interface.join("/", db.name, "db.yaml"),
            pytest.VERSION,
        )
        assert dataset.publication_owner == expected_publication_owner

    # repository
    assert dataset.repository == repository.name

    # repository_link : skipped for now

    # sampling_rates
    expected_sampling_rates = sorted(
        list(set([audiofile.sampling_rate(file) for file in db.files]))
    )
    assert dataset.sampling_rates == expected_sampling_rates

    # schemes
    expected_schemes = list(db.schemes)
    assert dataset.schemes == expected_schemes

    # schemes_table
    expected_schemes_table = [
        ["ID", "Dtype", "Min", "Labels", "Mappings"],
        ["age", "int", 0, "", ""],
        ["emotion", "str", "", "angry, happy, neutral", ""],
        ["gender", "str", "", "female, male", ""],
        ["speaker", "int", "", "0, 1", "age, gender"],
    ]
    assert dataset.schemes_table == expected_schemes_table

    # segment_durations
    expected_segment_durations = [0.5, 0.5, 150, 151]
    assert dataset.segment_durations == expected_segment_durations

    # segments
    expected_segments = str(len(db.segments))
    assert dataset.segments == expected_segments

    # short_description
    max_desc_length = 150
    expected_description = (
        db.description
        if (len(db.description) < max_desc_length)
        else f"{db.description[:max_desc_length - 3]}..."
    )
    assert dataset.short_description == expected_description

    # tables
    expected_tables = list(db)
    assert dataset.tables == expected_tables

    # tables_table
    expected_tables_table = [["ID", "Type", "Columns"]]
    for table_id in list(db):
        table = db[table_id]
        if isinstance(table, audformat.MiscTable):
            table_type = "misc"
        else:
            table_type = table.type
        columns = ", ".join(list(table.columns))
        expected_tables_table.append([table_id, table_type, columns])
    assert dataset.tables_table == expected_tables_table

    # version
    expected_version = pytest.VERSION
    assert dataset.version == expected_version


@pytest.mark.parametrize(
    "languages, iso_languages_expected",
    [
        (["greek", "Greek", "gr"], ["greek", "Greek", "gr"]),
        (["en", "English", "english", "En"], ["eng"]),
        (["de", "German", "german", "deu"], ["deu"]),
        (
            [
                "Algerian Arabic",
                "Egyptian Arabic",
                "Libyan Arabic",
                "Moroccan Arabic",
                "North Levantine Arabic",
            ],
            ["arz", "ary", "apc", "ayl", "arq"],
        ),
        (["Algerian Arabic"], ["arq"]),
        (["Egyptian Arabic"], ["arz"]),
        (["Libyan Arabic"], ["ayl"]),
        (["North Levantine Arabic"], ["apc"]),
        (["Moroccan Arabic"], ["ary"]),
    ],
)
def test_iso_language_mappings(languages, iso_languages_expected):
    """Test ISO 639-3 language mapping method."""
    iso_languages_calculated = audbcards.Dataset._map_iso_languages(languages)
    assert iso_languages_calculated == sorted(iso_languages_expected)


@pytest.mark.parametrize(
    "dbs",
    [
        ["minimal_db", "medium_db"],
    ],
)
def test_iso_language_property(dbs, cache, request):
    """Test ISO 639-3 language mapping property."""
    dbs = [request.getfixturevalue(db) for db in dbs]

    datasets = [
        audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache) for db in dbs
    ]
    _ = [dataset.iso_languages for dataset in datasets]


@pytest.mark.parametrize(
    "db",
    [
        "medium_db",
    ],
)
def test_dataset_example_media(db, cache, request):
    r"""Test Dataset.example_media.

    It checks that the desired audio file
    is selected as example.

    """
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache)

    # Relative path to audio file from database
    # as written in the dependencies table,
    # for example data/file.wav
    durations = [d.total_seconds() for d in db.files_duration(db.files)]
    median_duration = np.median([d for d in durations if 0.5 < d < 300])
    expected_example_index = min(
        range(len(durations)), key=lambda n: abs(durations[n] - median_duration)
    )
    expected_example = audeer.path(db.files[expected_example_index]).replace(
        os.sep, posixpath.sep
    )
    expected_example = "/".join(expected_example.split("/")[-2:])
    assert dataset.example_media == expected_example


@pytest.fixture
def constructor(tmpdir, medium_db, request):
    """Fixture to test Dataset constructor."""
    db = medium_db
    dataset_cache = audeer.mkdir(tmpdir, "cache")
    dataset_cache_filename = audbcards.Dataset._dataset_cache_path(
        db.name, pytest.VERSION, dataset_cache
    )

    ex0 = os.path.exists(dataset_cache_filename)

    ds_uncached = audbcards.Dataset(db.name, pytest.VERSION, cache_root=dataset_cache)

    ex1 = os.path.exists(dataset_cache_filename)

    ds_cached = audbcards.Dataset(db.name, pytest.VERSION, cache_root=dataset_cache)

    ex2 = os.path.exists(dataset_cache_filename)

    constructor = (ds_uncached, ds_cached, [ex0, ex1, ex2])

    return constructor


@pytest.mark.usefixtures("constructor")
class TestConstructor(object):
    """Test constructor class method.

    Testing of

    - existence of cache files
    - equality of property lists

    Currently the property values are not tested.
    Differences are unlikely.

    """

    def test_cache_file_existence(self, constructor):
        """Test that cache file comes into existence properly."""
        _, _, cache_file_existence = constructor
        expected_cache_file_existence = [False, True, True]
        assert cache_file_existence == expected_cache_file_existence

    def test_props_equal(self, constructor):
        """Cached and uncached datasets have equal props."""
        ds_uncached, ds_cached, _ = constructor
        props_uncached = ds_uncached._cached_properties()
        props_cached = ds_cached._cached_properties()
        list_props_uncached = list(props_uncached.keys())
        list_props_cached = list(props_cached.keys())
        assert list_props_uncached == list_props_cached


@pytest.mark.parametrize(
    "db",
    [
        "medium_db",
    ],
)
def test_dataset_cache_root(tmpdir, request, db):
    """Test configuration of cache root.

    ``cache_root`` can be provided by different options,
    in the following precedence:

    * as argument to ``audbcards.Dataset()``
    * as environment variable ``AUDBCARDS_CACHE_ROOT``
    * as ``audbcards.config.CACHE_ROOT`` entry

    """
    db = request.getfixturevalue(db)
    cache_root1 = audeer.mkdir(tmpdir, "cache1")
    cache_root2 = audeer.mkdir(tmpdir, "cache2")
    cache_root3 = audeer.mkdir(tmpdir, "cache3")
    assert audbcards.config.CACHE_ROOT == "~/.cache/audbcards"
    audbcards.config.CACHE_ROOT = cache_root1
    dataset = audbcards.Dataset(db.name, pytest.VERSION)
    assert dataset.cache_root == cache_root1
    os.environ["AUDBCARDS_CACHE_ROOT"] = cache_root2
    dataset = audbcards.Dataset(db.name, pytest.VERSION)
    assert dataset.cache_root == cache_root2
    dataset = audbcards.Dataset(
        db.name,
        pytest.VERSION,
        cache_root=cache_root3,
    )
    assert dataset.cache_root == cache_root3


def test_dataset_cache_path():
    """Test Value of default cache path."""
    cache_path_calculated = audbcards.core.dataset._Dataset._dataset_cache_path(
        "emodb",
        "1.2.1",
        "~/.cache/audbcards",
    )

    cache_path_expected = audeer.path(
        os.path.expanduser("~"),
        ".cache",
        "audbcards",
        "emodb",
        "1.2.1",
        "emodb-1.2.1.pkl",
    )
    assert cache_path_calculated == cache_path_expected


@pytest.mark.parametrize(
    "db",
    [
        "medium_db",
    ],
)
def test_dataset_cache_loading(audb_cache, tmpdir, repository, db, request):
    """Test cached properties after loading from cache.

    We no longer store all attributes/properties
    in cache as pickle files,
    but limit ourselves to the cached properties.
    This test ensures,
    that other attributes will be re-calculated.

    """
    db = request.getfixturevalue(db)
    cache_root = audeer.mkdir(tmpdir, "cache")
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache_root)
    del dataset
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache_root)
    deps = audb.dependencies(
        db.name,
        version=pytest.VERSION,
        cache_root=audb_cache,
    )
    backend_interface = repository.create_backend_interface()
    with backend_interface.backend:
        header = audb.info.header(
            db.name,
            version=pytest.VERSION,
            load_tables=True,
            cache_root=audb_cache,
        )
        # Compare only string representation,
        # as objects are not identical
        assert str(dataset.backend) == str(backend_interface)
        assert dataset.deps == deps
        # The dataset header is a not fully loaded `audformat.Database` object,
        # so we cannot directly use `audformat.Database.__eq__()`
        # to compare it.
        assert str(dataset.header) == str(header)
        assert dataset.repository_object == repository


@pytest.mark.parametrize(
    "text, expected",
    [
        ("abc\ndef", "abc\\ndef"),
        ("a" * 101, "a" * 97 + "..."),
        ('<a href="http://www.google.de">text link</a>', "text link"),
        (None, ""),
        (pd.NA, ""),
    ],
)
def test_dataset_parse_text(text, expected):
    """Test parsing of text."""
    assert audbcards.Dataset._parse_text(text) == expected


class TestDatasetLoadTables:
    r"""Test load_tables argument of audbcards.Dataset."""

    @classmethod
    @pytest.fixture(autouse=True)
    def prepare(cls, cache, medium_db):
        r"""Provide test class with cache, database name and database version.

        Args:
            cache: cache fixture
            medium_db: medium_db fixture

        """
        cls.name = medium_db.name
        cls.version = pytest.VERSION
        cls.cache_root = cache

    def assert_has_table_properties(self, expected: bool):
        r"""Assert dataset holds table related cached properties.

        Args:
            expected: if ``True``,
                ``dataset`` is expected to contain table related properties

        """
        table_related_properties = [
            "segment_durations",
            "segments",
        ]
        for table_related_property in table_related_properties:
            if expected:
                assert table_related_property in self.dataset.__dict__
            else:
                assert table_related_property not in self.dataset.__dict__

    def load_dataset(self, *, load_tables: bool):
        r"""Load dataset.

        Call ``audbcards.Dataset`` and assign result to ``self.dataset``.

        Args:
            load_tables: if ``True``,
                it caches properties,
                that need to load filewise/segmented tables

        """
        self.dataset = audbcards.Dataset(
            self.name,
            self.version,
            cache_root=self.cache_root,
            load_tables=load_tables,
        )

    @pytest.mark.parametrize("load_tables_first", [True, False])
    def test_load_tables(self, load_tables_first):
        r"""Load dataset with/without table related properties.

        This tests if the table related arguments
        are stored or omitted in cache,
        dependent on the ``load_tables`` argument.

        It also loads the dataset another two times from cache,
        with changing ``load_tables``
        arguments,
        which should always result
        in existing table related properties,
        as a cache stored first with ``load_tables=False``,
        should be updated when loading again with ``load_tables=True``.

        Args:
            load_tables_first: if ``True``,
                it calls ``audbcards.Dataset``
                with ``load_tables=True``
                during it first call

        """
        self.load_dataset(load_tables=load_tables_first)
        self.assert_has_table_properties(load_tables_first)
        self.load_dataset(load_tables=not load_tables_first)
        self.assert_has_table_properties(True)
        self.load_dataset(load_tables=load_tables_first)
        self.assert_has_table_properties(True)
