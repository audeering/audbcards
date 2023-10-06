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


def test_dataset(cache, tmpdir, db):
    dataset_cache = audeer.mkdir(audeer.path(tmpdir, 'cache'))
    dataset = audbcards.Dataset(
        pytest.NAME,
        pytest.VERSION,
        cache_root=dataset_cache,
    )

    # __init__
    assert dataset.name == pytest.NAME
    assert dataset.version == pytest.VERSION
    assert dataset._repository == pytest.REPOSITORY
    expected_header = audb.info.header(
        pytest.NAME,
        version=pytest.VERSION,
        cache_root=cache,
    )
    assert str(dataset.header) == str(expected_header)
    expected_deps = audb.dependencies(
        pytest.NAME,
        version=pytest.VERSION,
        cache_root=cache,
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

    # example
    # Relative path to audio file from database
    # as written in the dependencies table,
    # for example data/file.wav
    durations = [d.total_seconds() for d in db.files_duration(db.files)]
    median_duration = np.median([d for d in durations if 0.5 < d < 300])
    expected_example_index = min(
        range(len(durations)),
        key=lambda n: abs(durations[n] - median_duration)
    )
    expected_example = audeer.path(
        db.files[expected_example_index]
    ).replace(os.sep, posixpath.sep)
    expected_example = '/'.join(expected_example.split('/')[-2:])
    assert dataset.example == expected_example

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
    expected_repository = pytest.REPOSITORY.name
    assert dataset.repository == expected_repository

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
    expected_schemes = audbcards.core.dataset.format_schemes(db.schemes)
    assert dataset.schemes == expected_schemes

    # short_description
    max_desc_length = 150
    expected_description = db.description if (
        len(db.description) < max_desc_length
    ) else f'{db.description[:max_desc_length - 3]}...'
    assert dataset.short_description == expected_description

    # version
    expected_version = pytest.VERSION
    assert dataset.version == expected_version


@pytest.mark.parametrize(
    'scheme_names, scheme_dtypes, labels, expected',
    [
        (
            ['emotion', 'age', 'gender', 'language', 'speaker'],
            ['str', 'int', 'str', 'str', 'int'],
            [
                ['happy', 'sad'],
                None,
                ['female', 'male'],
                ['DE', 'EN'],
                'speaker',
            ],
            'emotion: [happy, sad], speaker: [age, gender, language], '
            'age, gender, language'
        ),
        (
            ['emotion', 'age', 'gender', 'speaker', 'audio_quality'],
            ['str', 'int', 'str', 'int', 'str'],
            [
                ['happy', 'sad'],
                None,
                ['female', 'male'],
                'speaker',
                ['good', 'bad']
            ],
            'emotion: [happy, sad], speaker: [age, gender], '
            'age, audio_quality, gender'
        ),
    ]
)
def test_format_schemes(scheme_names, scheme_dtypes, labels, expected):
    # Init database to contain schemes
    db = audformat.Database(name=pytest.NAME)
    for i, scheme_name in enumerate(scheme_names):
        # Create actual schemes
        if scheme_name == 'speaker':
            db['speaker'] = audformat.MiscTable(pd.Index(
                [0],
                dtype='Int8',
                name='speaker'
            ))
            if 'age' in scheme_names:
                db['speaker']['age'] = audformat.Column(scheme_id='age')
            if 'gender' in scheme_names:
                db['speaker']['gender'] = audformat.Column(scheme_id='gender')
            if 'language' in scheme_names:
                db['speaker']['language'] = audformat.Column(
                    scheme_id='language'
                )
        db.schemes[scheme_name] = audformat.Scheme(
            dtype=scheme_dtypes[i],
            labels=labels[i],
        )
    # Generate scheme str with format_scheme()
    scheme_str = audbcards.core.dataset.format_schemes(db.schemes)
    assert scheme_str == expected


@pytest.mark.parametrize(
    'sample',
    [
        ['a', 'b', 'c', 'd', 'e']
    ]
)
@pytest.mark.parametrize(
    'limit, replacement_text, expected',
    [
        (2, '...', ['a', '...', 'e']),
        (2, '###', ['a', '###', 'e']),
        (4, '...', ['a', 'b', '...', 'd', 'e']),
    ]
)
def test_limit_presented_samples(sample, limit, replacement_text, expected):
    limited_sample = audbcards.core.dataset.limit_presented_samples(
        sample, limit, replacement_text
    )
    assert limited_sample == expected
