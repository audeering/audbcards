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


@pytest.mark.parametrize(
    "db",
    [
        "bare_db",
        "minimal_db",
        "medium_db",
    ],
)
def test_datacard(db, cache, request):
    """Test datacard creation from jinja2 templates."""
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache)
    datacard = audbcards.Datacard(dataset)
    content = datacard._render_template()
    content = content.rstrip()
    expected_content = load_rendered_template(db.name)

    # Remove lines that depend on author/local machine
    for pattern in [
        re.compile("^published.*$", flags=(re.MULTILINE)),
        re.compile("^repository.*$", flags=(re.MULTILINE)),
        re.compile("^license.*$", flags=(re.MULTILINE)),
    ]:
        content = re.sub(pattern, "", content)
        expected_content = re.sub(pattern, "", expected_content)

    assert content == expected_content


@pytest.mark.parametrize(
    "db",
    [
        "medium_db",
    ],
)
def test_datacard_example_media(db, cache, request):
    r"""Test Datacard.example_media.

    It checks that the desired audio file
    is selected as example.

    """
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache)
    datacard = audbcards.Datacard(dataset)

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
    assert datacard.example_media == expected_example


@pytest.mark.parametrize(
    "db, expected_min, expected_max",
    [
        ("bare_db", 0, 0),
        ("medium_db", 1, 301),
    ],
)
def test_datacard_file_duration_distribution(
    tmpdir,
    cache,
    request,
    db,
    expected_min,
    expected_max,
):
    r"""Test the Datacard.file_duration_distribution.

    It checks if the desired distribution PNG file is created,
    and the expected RST string
    to include the distribution is returned.

    """
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache)

    datacard_path = audeer.mkdir(tmpdir, "datasets")
    datacard = audbcards.Datacard(dataset, path=datacard_path)

    # Without specifying sphinx src and build dirs,
    # we do not expect a PNG file
    distribution_str = datacard.file_duration_distribution
    build_dir = audeer.mkdir(tmpdir, "build", "html")
    src_dir = audeer.mkdir(tmpdir, "docs")
    image_file = audeer.path(
        build_dir,
        datacard.path,
        db.name,
        f"{db.name}-file-durations.png",
    )
    assert not os.path.exists(image_file)
    expected_distribution_str = f"{expected_min:.1f} s .. {expected_max:.1f} s"
    assert expected_distribution_str == distribution_str

    # Set sphinx src and build dir and execute again
    datacard.sphinx_build_dir = build_dir
    datacard.sphinx_src_dir = src_dir
    distribution_str = datacard.file_duration_distribution
    assert os.path.exists(image_file)
    expected_distribution_str = (
        f"{expected_min:.1f} s |{db.name}-file-durations| {expected_max:.1f} s"
    )
    assert expected_distribution_str == distribution_str


@pytest.mark.parametrize(
    "db",
    [
        "medium_db",
    ],
)
def test_datacard_player(tmpdir, db, cache, request):
    r"""Test the Datacard.player.

    It checks if the desired waveplot PNG file is created,
    the example audio file is copied to the build folder,
    and the expected RST string
    to include the player is returned.

    """
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache)

    datacard_path = audeer.mkdir(tmpdir, "datasets")
    datacard = audbcards.Datacard(dataset, path=datacard_path)

    # Execute player
    # without specifying sphinx src and build dirs
    player_str = datacard.player()
    build_dir = audeer.mkdir(tmpdir, "build", "html")
    src_dir = audeer.mkdir(tmpdir, "docs")
    media_file = audeer.path(
        build_dir,
        datacard.path,
        db.name,
        datacard.example_media,
    )
    image_file = audeer.path(
        src_dir,
        datacard.path,
        db.name,
        f"{db.name}.png",
    )
    assert not os.path.exists(media_file)
    assert not os.path.exists(image_file)

    # Set sphinx src and build dir and execute again
    datacard.sphinx_build_dir = build_dir
    datacard.sphinx_src_dir = src_dir
    player_str = datacard.player(datacard.example_media)
    assert os.path.exists(media_file)
    assert os.path.exists(image_file)

    # Expected waveform plot
    signal, sampling_rate = audiofile.read(
        media_file,
        always_2d=True,
    )
    plt.figure(figsize=[3, 0.5])
    ax = plt.subplot(111)
    audplot.waveform(signal[0, :], ax=ax)
    set_plot_margins()
    outfile = f"{os.path.join(cache, db.name)}.png"
    plt.savefig(outfile)
    plt.close()
    expected_waveform = open(outfile, "rb").read()
    # Check if generated images are exactly the same (pixel-wise)
    waveform = open(image_file, "rb").read()
    assert waveform == expected_waveform

    # Append audio to the expected player_str
    expected_player_str = (
        f".. image:: ./{db.name}/{db.name}.png\n"
        "\n"
        ".. raw:: html\n"
        "\n"
        f'    <p><audio controls src="./{db.name}/{datacard.example_media}">'
        f"</audio></p>"
    )
    # Check if the generated player_str and the expected matches
    assert expected_player_str == player_str


@pytest.mark.parametrize(
    "dbs",
    [
        ["minimal_db", "medium_db"],
    ],
)
def test_create_datasets_page(tmpdir, dbs, cache, request):
    r"""Test the creation of an RST file with an datasets overview table."""
    dbs = [request.getfixturevalue(db) for db in dbs]
    datasets = [
        audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache) for db in dbs
    ]
    rst_file = audeer.path(tmpdir, "datasets_page.rst")
    create_datasets_page(datasets, rst_file=rst_file)
    assert os.path.exists(rst_file)
    assert os.path.exists(audeer.replace_file_extension(rst_file, "csv"))


def load_rendered_template(name: str) -> str:
    r"""Load the expected rendered RST file."""
    fpath = os.path.join(pytest.TEMPLATE_DIR, f"{name}.rst")
    with open(fpath, "r") as file:
        rendered_rst = file.read().rstrip()
    return rendered_rst
