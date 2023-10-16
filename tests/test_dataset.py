import pandas as pd
import pytest

import audb
import audeer
import audformat
import audiofile

import audbcards


@pytest.mark.parametrize(
    'db',
    [
        'medium_db',
    ],
)
def test_dataset(audb_cache, tmpdir, repository, db, request):
    r"""Test audbcards.Dataset object and all its properties."""
    db = request.getfixturevalue(db)

    dataset_cache = audeer.mkdir(audeer.path(tmpdir, 'cache'))
    dataset = audbcards.Dataset(
        db.name,
        pytest.VERSION,
        cache_root=dataset_cache,
    )

    # __init__
    assert dataset.name == db.name
    assert dataset.version == pytest.VERSION
    assert dataset._repository == repository
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
    expected_archives = len(
        expected_df.loc[expected_deps.media].archive.unique()
    )
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
        list(
            set(
                [
                    audiofile.channels(file)
                    for file in db.files
                ]
            )
        )
    )
    assert dataset.channels == expected_channels

    # duration
    expected_duration = db.files_duration(db.files).sum()
    assert dataset.duration == expected_duration

    # files
    expected_files = len(db.files)
    assert dataset.files == expected_files

    # formats
    expected_formats = sorted(
        list(
            set(
                [
                    audeer.file_extension(file)
                    for file in db.files
                ]
            )
        )
    )
    assert dataset.formats == expected_formats

    # license
    expected_license = db.license or 'Unknown'
    assert dataset.license == expected_license

    # license link
    if (
            db.license_url is None
            or len(db.license_url) == 0
    ):
        expected_license_link = None
    else:
        expected_license_link = db.license_url
    assert dataset.license_link == expected_license_link

    # publication: skipped for now

    # repository
    assert dataset.repository == repository.name

    # repository_link : skipped for now

    # sampling_rates
    expected_sampling_rates = sorted(
        list(
            set(
                [
                    audiofile.sampling_rate(file)
                    for file in db.files
                ]
            )
        )
    )
    assert dataset.sampling_rates == expected_sampling_rates

    # schemes
    expected_schemes = list(db.schemes)
    assert dataset.schemes == expected_schemes

    # schemes_table
    expected_schemes_table = [
        ['ID', 'Dtype', 'Min', 'Labels', 'Mappings'],
        ['age', 'int', 0, '', ''],
        ['emotion', 'str', '', 'angry, happy, neutral', ''],
        ['gender', 'str', '', 'female, male', ''],
        ['speaker', 'int', '', '0, 1', 'age, gender'],
    ]
    assert dataset.schemes_table == expected_schemes_table

    # short_description
    max_desc_length = 150
    expected_description = db.description if (
        len(db.description) < max_desc_length
    ) else f'{db.description[:max_desc_length - 3]}...'
    assert dataset.short_description == expected_description

    # tables
    expected_tables = list(db)
    assert dataset.tables == expected_tables

    # tables_table
    expected_tables_table = [['ID', 'Type', 'Columns']]
    for table_id in list(db):
        table = db[table_id]
        if isinstance(table, audformat.MiscTable):
            table_type = 'misc'
        else:
            table_type = table.type
        columns = ', '.join(list(table.columns))
        expected_tables_table.append([table_id, table_type, columns])
    assert dataset.tables_table == expected_tables_table

    # version
    expected_version = pytest.VERSION
    assert dataset.version == expected_version
