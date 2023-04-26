import os

import pytest

import audeer
import audiofile
import audb
import audbcards
import audplot

import numpy as np
import matplotlib.pyplot as plt


def test_dataset(db):
    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION)

    # __init__
    assert dataset.name == pytest.NAME
    assert dataset.version == pytest.VERSION
    assert dataset.repository == pytest.REPOSITORY
    expected_header = audb.info.header(
        pytest.NAME,
        version=pytest.VERSION,
    )
    assert str(dataset.header) == str(expected_header)
    expected_deps = audb.dependencies(
        pytest.NAME,
        version=pytest.VERSION,
    )
    # pd.testing.assert_frame_equal(dataset.deps(), expected_deps())

    # archives
    assert dataset.archives == str(len(db.files))

    # bit depths
    expected_bit_depths = set([
        str(audiofile.bit_depth(file)) for file in db.files
        if audiofile.bit_depth(file)
    ])
    assert dataset.bit_depths == ', '.join(expected_bit_depths)

    # channels
    expected_channels = set(
        [str(audiofile.channels(file)) for file in db.files]
    )
    assert dataset.channels == ', '.join(expected_channels)

    # duration
    expected_duration = db.files_duration(db.files).sum()
    assert dataset.duration == str(expected_duration)

    # example
    durations = [d.total_seconds() for d in db.files_duration(db.files)]
    median_duration = np.median([d for d in durations if 0.5 < d < 300])
    expected_example_index = min(
        range(len(durations)),
        key=lambda n: abs(durations[n] - median_duration)
    )
    expected_example = db.files[expected_example_index].split('/')[-2:]
    assert dataset.example == '/'.join(expected_example)

    # files
    assert dataset.files == str(len(db.files))

    # formats
    expected_formats = set([
        audeer.file_extension(file) for file in db.files
    ])
    assert dataset.formats == ', '.join(expected_formats)

    # license link
    license = db.license or 'Unknown'
    if (
            db.license_url is None
            or len(db.license_url) == 0
    ):
        expected_license_link = license
    else:
        expected_license_link = f'`{license} <{db.license_url}>`__'
    assert dataset.license_link == expected_license_link

    # name_link
    expected_name_link = f'`{db.name} <./datasets/{db.name}.html>`__'
    assert dataset.name_link == expected_name_link

    # publication: skipped for now
    # repository_link : skipped for now

    # sampling_rates
    expected_sampling_rates = set([
        str(audiofile.sampling_rate(file)) for file in db.files
    ])
    assert dataset.sampling_rates == ', '.join(expected_sampling_rates)

    # schemes
    expected_schemes = audbcards.core.dataset.format_schemes(db.schemes)
    assert dataset.schemes == expected_schemes

    # short_description
    max_desc_length = 150
    expected_description = db.description if (
            len(db.description) < max_desc_length
    ) else f'{db.description[:max_desc_length - 3]}...'
    assert dataset.short_description == expected_description

    # version_link
    github = 'https://github.com/audeering'
    branch = 'main'
    url = f'{github}/{db.name}/blob/{branch}/CHANGELOG.md'
    expected_version_link = f'`{pytest.VERSION} <{url}>`__'
    assert dataset.version_link == expected_version_link

