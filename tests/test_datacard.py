import os
import re

import matplotlib.pyplot as plt
import pytest

import audeer
import audiofile
import audplot

import audbcards
from audbcards.core.dataset import create_datasets_page
from audbcards.core.utils import set_plot_margins


BUILD = audeer.path('..', 'build', 'html')


def test_datacard_lines_similar(db, default_template, cache):
    """Create datacard using jinja2 via Dataset and Datacard.

    The assertions for exact identity are currently too strict.
    Therefore this uses text similarities as obtained from
    the difflib builtins. These are based on

    - average (or rather median and mean) similarities per line
    - percentage of lines differing between original and rendered

    """
    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION, cache)
    dc = audbcards.Datacard(dataset)
    content = dc._render_template()
    content = content.rstrip()

    # Remove lines that depend on author/local machine
    for pattern in [
            re.compile('^published.*$', flags=(re.MULTILINE)),
            re.compile('^repository.*$', flags=(re.MULTILINE)),
    ]:
        content = re.sub(pattern, '', content)
        default_template = re.sub(pattern, '', default_template)

    assert content == default_template


def test_datacard_player(db, cache):

    dataset = audbcards.Dataset(pytest.NAME, pytest.VERSION, cache)
    datacard = audbcards.Datacard(dataset)

    player_str = datacard.player(dataset.example)

    # Check if file has been copied under the build folder
    dst_dir = f'{BUILD}/datasets/{db.name}'
    assert os.path.exists(os.path.join(dst_dir, dataset.example))

    # Expected waveform plot
    signal, sampling_rate = audiofile.read(
        os.path.join(dst_dir, dataset.example),
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
        f'.. image:: ../{db.name}.png\n'
        '\n'
        '.. raw:: html\n'
        '\n'
        f'    <p><audio controls src="{db.name}/{dataset.example}">'
        f'</audio></p>'
    )
    # Check if the generated player_str and the expected matches
    assert expected_player_str == player_str


def test_create_datasets_page(db):

    datasets = [audbcards.Dataset(pytest.NAME, pytest.VERSION)] * 4
    create_datasets_page(datasets, ofbase="datasets_page")
