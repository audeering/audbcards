import os

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


CACHE = './cache'
BUILD = audeer.path('..', 'build', 'html')


def test_dataset(tmpdir, db):
    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION)

    # __init__
    assert dataset.name == pytest.NAME
    assert dataset.version == pytest.VERSION
    assert dataset.repository == pytest.REPOSITORY
    expected_header = audb.info.header(
        pytest.NAME,
        version=pytest.VERSION,
        cache_root=CACHE,
    )
    assert str(dataset.header) == str(expected_header)
    expected_deps = audb.dependencies(
        pytest.NAME,
        version=pytest.VERSION,
        cache_root=CACHE,
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
    expected_example = '/'.join(
        db.files[expected_example_index].split('/')[-2:]
    )
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

    # Test create_datacard_page()
    # Create datacard page using developed method
    # Requires `datasets` folder to exist
    audeer.mkdir('./datasets')
    audbcards.core.dataset.create_datacard_page(dataset)
    # Generate expected page to use for comparison
    rst_file = f'datasets/expected_{pytest.NAME}.rst'
    with open(rst_file, 'w') as fp:
        # Link to page
        fp.write(f'.. _{pytest.NAME}:\n')
        fp.write('\n')

        # Heading
        fp.write(f'{pytest.NAME}\n')
        fp.write('-' * len(pytest.NAME))
        fp.write('\n\n')

        # Author
        if db.author is not None:
            fp.write(f'Created by {db.author}\n')
            fp.write('\n\n')

        # Overview table
        fp.write('============= ======================\n')
        fp.write(f'version       {expected_version_link}\n')
        fp.write(f'license       {expected_license_link}\n')
        fp.write(f'source        {db.source}\n')
        fp.write(f'usage         {db.usage}\n')
        if db.languages is not None:
            fp.write(f'languages     {", ".join(db.languages)}\n')
        fp.write(f'format        {expected_formats}\n')
        fp.write(f'channel       {expected_channels}\n')
        fp.write(f'sampling rate {expected_sampling_rates}\n')
        fp.write(f'bit depth     {expected_bit_depths}\n')
        fp.write(f'duration      {expected_duration}\n')
        fp.write(f'files         {expected_files}\n')
        # repository_link and publication are currently
        # ingested from the dataset object as we're not
        # testing them separately for now
        fp.write(f'repository    {dataset.repository_link}\n')
        fp.write(f'published     {dataset.publication}\n')
        fp.write('============= ======================\n')
        fp.write('\n\n')

        # Description
        if (
                db.description is not None
                and len(db.description) > 0
        ):
            description = db.description.replace('|', r'\|')
            fp.write('Description\n')
            fp.write('^^^^^^^^^^^\n')
            fp.write('\n')
            fp.write(description)
            fp.write('\n\n')

        # Audio example
        file = expected_example
        if len(file) > 0:
            fp.write('Example\n')
            fp.write('^^^^^^^\n')
            fp.write('\n')
            fp.write(f':file:`{file}`\n')
            fp.write('\n')
            fp.write(f'{expected_player_str}\n')
            fp.write('\n')

        # Tables
        tables = list(db)
        types = []
        for table_id in tables:
            table = db[table_id]
            if isinstance(table, audformat.MiscTable):
                types.append('misc')
            else:
                types.append(table.type)
        columns = [list(db[table_id].columns) for table_id in tables]
        fp.write('Tables\n')
        fp.write('^^^^^^\n')
        fp.write('\n')
        fp.write('.. csv-table::\n')
        fp.write('   :header: ID,Type,Columns\n')
        fp.write('   :widths: 20, 10, 70\n')
        fp.write('\n')
        for table, type_, column in zip(tables, types, columns):
            fp.write(f'    "{table}", "{type_}", "{", ".join(column)}"\n')
        fp.write('\n\n')

        # Schemes
        if len(db.schemes) > 0:
            has_minimums = any(
                [db.schemes[s].minimum is not None for s in db.schemes]
            )
            has_maximums = any(
                [db.schemes[s].maximum is not None for s in db.schemes]
            )
            has_labels = any(
                [db.schemes[s].labels is not None for s in db.schemes]
            )
            has_mappings = any(
                [
                    isinstance(db.schemes[s].labels, (str, dict))
                    for s in db.schemes
                ]
            )
            header_line = '   :header: ID,Dtype'
            if has_minimums:
                header_line += ',Min'
            if has_maximums:
                header_line += ',Max'
            if has_labels:
                header_line += ',Labels'
            if has_mappings:
                header_line += ',Mappings'
            header_line += '\n'
            fp.write('Schemes\n')
            fp.write('^^^^^^^\n')
            fp.write('\n')
            fp.write('.. csv-table::\n')
            fp.write(header_line)
            fp.write('\n')
            for scheme_id in db.schemes:
                fp.write(f'    "{scheme_id}", ')
                scheme = db.schemes[scheme_id]
                fp.write(f'"{scheme.dtype}"')
                if has_minimums:
                    minimum = scheme.minimum or ''
                    fp.write(f', "{minimum}"')
                if has_maximums:
                    maximum = scheme.maximum or ''
                    fp.write(f', "{maximum}"')
                if has_labels:
                    if scheme.labels is None:
                        labels = []
                    else:
                        labels = sorted(scheme._labels_to_list())
                        labels = [str(label) for label in labels]
                        # Avoid `_` at end of label,
                        # as this has special meaning in RST (link)
                        labels = [
                            label[:-1] + r'\_'
                            if label.endswith('_')
                            else label
                            for label in labels
                        ]
                        labels = audbcards.core.dataset.limit_presented_samples(
                            labels,
                            15,
                            replacement_text='[...]',
                        )
                    fp.write(f', "{", ".join(labels)}"')
                if has_mappings:
                    if not isinstance(scheme.labels, (str, dict)):
                        mappings = ''
                    else:
                        labels = scheme._labels_to_dict()
                        # Mappings can contain a single mapping
                        # or a deeper nestings.
                        # In the first case we just present ✓,
                        # in the second case the keys of the nested dict.
                        # {'f': 'female', 'm': 'male'}
                        # or
                        # {'s1': {'gender': 'male', 'age': 21}}
                        mappings = list(labels.values())
                        if isinstance(mappings[0], dict):
                            # e.g. {'s1': {'gender': 'male', 'age': 21}}
                            mappings = sorted(list(mappings[0].keys()))
                            mappings = f'{", ".join(mappings)}'
                        else:
                            # e.g. {'f': 'female', 'm': 'male'}
                            mappings = '✓'
                        fp.write(f', "{mappings}"')
                fp.write('\n')

    # Check if generated rst files are exactly the same
    assert open(rst_file, 'rb').read(
    ) == open(f'datasets/{dataset.name}.rst', 'rb').read()
