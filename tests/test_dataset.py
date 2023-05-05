import os
import posixpath

import pytest

import audeer
import audiofile
import audb
import audformat
import audbcards
import audplot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BUILD = audeer.path('..', 'build', 'html')


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
    assert dataset.repository == pytest.REPOSITORY
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
    pd.testing.assert_frame_equal(dataset.deps(), expected_deps())

    # archives
    assert dataset.archives == str(len(db.files))

    # bit depths
    expected_bit_depths = ', '.join(set([
        str(audiofile.bit_depth(file)) for file in db.files
        if audiofile.bit_depth(file)
    ]))
    assert dataset.bit_depths == expected_bit_depths

    # channels
    expected_channels = ', '.join(set(
        [str(audiofile.channels(file)) for file in db.files]
    ))
    assert dataset.channels == expected_channels

    # duration
    expected_duration = str(db.files_duration(db.files).sum())
    assert dataset.duration == expected_duration

    # example
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
    expected_files = str(len(db.files))
    assert dataset.files == expected_files

    # formats
    expected_formats = ', '.join(set([
        audeer.file_extension(file) for file in db.files
    ]))
    assert dataset.formats == expected_formats

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
    expected_sampling_rates = ', '.join(set([
        str(audiofile.sampling_rate(file)) for file in db.files
    ]))
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

    # version_link
    github = 'https://github.com/audeering'
    branch = 'main'
    url = f'{github}/{db.name}/blob/{branch}/CHANGELOG.md'
    expected_version_link = f'`{pytest.VERSION} <{url}>`__'
    assert dataset.version_link == expected_version_link

    # Test player
    # Init player using dataset object
    plot_waveform = True
    player_str = dataset.player(dataset.example, waveform=plot_waveform)
    # Generate expected player
    expected_player_str = ''
    # Check if file has been copied under the build folder
    dst_dir = f'{BUILD}/datasets/{db.name}'
    assert os.path.exists(os.path.join(dst_dir, expected_example))
    # Add plot of waveform
    if plot_waveform:
        signal, sampling_rate = audiofile.read(
            os.path.join(dst_dir, expected_example),
            always_2d=True,
        )
        plt.figure(figsize=[3, .5])
        ax = plt.subplot(111)
        audplot.waveform(signal[0, :], ax=ax)
        audbcards.core.dataset.set_plot_margins()
        plt.savefig(f'{os.path.join(tmpdir, db.name)}.png')
        plt.close()
        # Check if generated images are exactly the same (pixel-wise)
        assert open(f'{db.name}.png', 'rb').read(
        ) == open(f'{os.path.join(tmpdir, db.name)}.png', 'rb').read()
        # Append image to the expected player_str
        expected_player_str += (
            f'.. image:: ../{db.name}.png\n'
            '\n'
        )
    # Append audio to the expected player_str
    expected_player_str += (
        '.. raw:: html\n'
        '\n'
        f'    <p><audio controls src="{db.name}/{expected_example}">'
        f'</audio></p>'
    )
    # Check if the generated player_str and the expected matches
    assert expected_player_str == player_str


@pytest.mark.parametrize(
    'scheme_names, scheme_dtypes, labels',
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
        ),
    ]
)
def test_format_schemes(scheme_names, scheme_dtypes, labels):
    expected_scheme_str_dict = {
        scheme: f'{scheme}: [' for scheme in scheme_names
    }
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
            db['speaker']['age'] = audformat.Column(scheme_id='age')
            db['speaker']['gender'] = audformat.Column(scheme_id='gender')
            db['speaker']['language'] = audformat.Column(scheme_id='language')
        db.schemes[scheme_name] = audformat.Scheme(
            dtype=scheme_dtypes[i],
            labels=labels[i],
        )
        # Generate expected formatted scheme string
        if scheme_name == 'emotion':
            for label in labels[i]:
                expected_scheme_str_dict[scheme_name] += f'{label}, '
            expected_scheme_str_dict[
                scheme_name
            ] = f'{expected_scheme_str_dict[scheme_name][:-2]}], '
        elif scheme_name == 'speaker':
            for speaker_scheme in scheme_names:
                if speaker_scheme not in ['speaker', 'emotion']:
                    expected_scheme_str_dict[
                        scheme_name
                    ] += f'{speaker_scheme}, '
            expected_scheme_str_dict[
                scheme_name
            ] = f'{expected_scheme_str_dict[scheme_name][:-2]}], '

        else:
            expected_scheme_str_dict[scheme_name] = f'{scheme_name}, '
    # Construct the string object
    scheme_names.remove('speaker')
    scheme_names.insert(1, 'speaker')
    expected_scheme_str = ''
    for scheme_name in scheme_names:
        expected_scheme_str += expected_scheme_str_dict[scheme_name]
    # Remove the ", " at the end
    expected_scheme_str = expected_scheme_str[:-2]
    # Generate scheme str with format_scheme()
    scheme_str = audbcards.core.dataset.format_schemes(db.schemes)
    assert scheme_str == expected_scheme_str


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
