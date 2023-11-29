import os
import posixpath
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

import audeer
import audiofile
import audplot

import audbcards
from audbcards.core.dataset import create_datasets_page
from audbcards.core.utils import set_plot_margins


BUILD = audeer.path('..', 'build', 'html')


@pytest.mark.parametrize(
    'db',
    [
        'bare_db',
        'minimal_db',
        'medium_db',
    ],
)
def test_datacard(db, cache, request):
    """Test datacard creation from jinja2 templates."""
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache)
    datacard = audbcards.Datacard(dataset)
    content = datacard._render_template()
    content = content.rstrip()
    expected_content = load_rendered_template(db.name)

    # Remove lines that depend on author/local machine
    for pattern in [
            re.compile('^published.*$', flags=(re.MULTILINE)),
            re.compile('^repository.*$', flags=(re.MULTILINE)),
    ]:
        content = re.sub(pattern, '', content)
        expected_content = re.sub(pattern, '', expected_content)

    assert content == expected_content


@pytest.mark.parametrize(
    'db',
    [
        'medium_db',
    ],
)
def test_datacard_example(db, cache, request):
    r"""Test Datacard.example.

    It checks that the desired audio file
    is selected as example.

    """
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache)
    datacard = audbcards.Datacard(dataset)

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
    assert datacard.example == expected_example


@pytest.mark.parametrize(
    'db',
    [
        'medium_db',
    ],
)
def test_datacard_player(db, cache, request):
    r"""Test the Datacard.player.

    It checks if the desired waveplot PNG file is created,
    the example audio file is copied to the build folder,
    and the expected RST string
    to include the player is returned.

    """
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache)
    datacard = audbcards.Datacard(dataset)

    player_str = datacard.player(datacard.example)

    # Check if file has been copied under the build folder
    dst_dir = f'{BUILD}/datasets/{db.name}'
    assert os.path.exists(os.path.join(dst_dir, datacard.example))

    # Expected waveform plot
    signal, sampling_rate = audiofile.read(
        os.path.join(dst_dir, datacard.example),
        always_2d=True,
    )
    plt.figure(figsize=[3, .5])
    ax = plt.subplot(111)
    audplot.waveform(signal[0, :], ax=ax)
    set_plot_margins()
    outfile = f'{os.path.join(cache, db.name)}.png'
    plt.savefig(outfile)
    plt.close()
    expected_waveform = open(outfile, 'rb').read()
    # Check if generated images are exactly the same (pixel-wise)
    waveform = open(f'{db.name}.png', 'rb').read()
    assert waveform == expected_waveform

    # Append audio to the expected player_str
    expected_player_str = (
        f'.. image:: ./{db.name}/{db.name}.png\n'
        '\n'
        '.. raw:: html\n'
        '\n'
        f'    <p><audio controls src="./{db.name}/{datacard.example}">'
        f'</audio></p>'
    )
    # Check if the generated player_str and the expected matches
    assert expected_player_str == player_str


@pytest.mark.parametrize(
    'dbs',
    [
        ['minimal_db', 'medium_db'],
    ],
)
def test_create_datasets_page(dbs, request):
    r"""Test the creation of an RST file with an datasets overview table."""
    dbs = [request.getfixturevalue(db) for db in dbs]
    datasets = [audbcards.Dataset(db.name, pytest.VERSION) for db in dbs]
    create_datasets_page(datasets, rst_file="datasets_page.rst")


def load_rendered_template(name: str) -> str:
    r"""Load the expected rendered RST file."""
    fpath = os.path.join(pytest.TEMPLATE_DIR, f'{name}.rst')
    with open(fpath, 'r') as file:
        rendered_rst = file.read().rstrip()
    return rendered_rst
