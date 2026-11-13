import csv
import json
import os
import re

import jinja2
import matplotlib.pyplot as plt
import pytest

import audb
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
        "json_db",
        "mixed_db",
    ],
)
def test_datacard(tmpdir, db, cache, request):
    """Test datacard creation from jinja2 templates."""
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache)
    datacard = audbcards.Datacard(dataset, cache_root=cache)

    # Set sphinx src and build dir
    build_dir = audeer.mkdir(tmpdir, "build", "html")
    src_dir = audeer.mkdir(tmpdir, "docs")
    datacard.sphinx_build_dir = build_dir
    datacard.sphinx_src_dir = src_dir

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
        "json_db",
        "mixed_db",
    ],
)
@pytest.mark.parametrize("example", [True, False])
def test_datacard_example(tmpdir, db, example, cache, request):
    r"""Test example section is only included if example is True."""
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache)
    datacard = audbcards.Datacard(dataset, example=example, cache_root=cache)
    datacard.sphinx_build_dir = audeer.mkdir(tmpdir, "build", "html")
    datacard.sphinx_src_dir = audeer.mkdir(tmpdir, "docs")

    content = datacard._render_template()

    lines = content.split("\n")
    assert ("Example" in lines) == example
    if dataset.example_media is not None:
        assert (f":file:`{dataset.example_media}`" in content) == example
    if dataset.example_json is not None:
        assert (f":file:`{dataset.example_json}`" in content) == example
        assert (".. code:: json" in content) == example


@pytest.mark.parametrize("db", ["medium_db", "json_db"])
def test_datacard_example_custom_template(tmpdir, db, cache, request):
    r"""Test example section is hidden for user templates.

    A user provided ``datacard_example.j2``
    that does not check ``example``
    should not render the example section
    when ``example`` is ``False``.

    """
    db = request.getfixturevalue(db)
    template_dir = audeer.mkdir(tmpdir, "templates")
    with open(os.path.join(template_dir, "datacard_example.j2"), "w") as fp:
        fp.write(
            "{% if example_media is not none %}\n"
            "Custom media: {{ example_media }}\n"
            "{% endif %}\n"
            "{% if example_json is not none %}\n"
            "Custom json: {{ example_json }}\n"
            "{% endif %}\n"
        )
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache)
    datacard = audbcards.Datacard(
        dataset,
        example=False,
        template_dir=template_dir,
        cache_root=cache,
    )

    content = datacard._render_template()

    assert "Custom media" not in content
    assert "Custom json" not in content


@pytest.mark.parametrize("db", ["minimal_db", "medium_db"])
@pytest.mark.parametrize("table_preview", [True, False])
def test_datacard_table_preview(tmpdir, db, table_preview, cache, request):
    r"""Test table preview is only included if table_preview is True."""
    db = request.getfixturevalue(db)
    dataset = audbcards.Dataset(db.name, pytest.VERSION, cache_root=cache)
    datacard = audbcards.Datacard(
        dataset,
        table_preview=table_preview,
        cache_root=cache,
    )
    datacard.sphinx_build_dir = audeer.mkdir(tmpdir, "build", "html")
    datacard.sphinx_src_dir = audeer.mkdir(tmpdir, "docs")

    content = datacard._render_template()

    # Tables overview is always shown
    assert "Tables\n^^^^^^" in content
    for table in db.tables:
        assert f"<td><p>{table}</p></td>" in content
    # Preview and clickable rows are only shown with table_preview
    assert ("Click on a row to toggle a preview." in content) == table_preview
    assert ('onClick="toggleRow(this)"' in content) == table_preview
    assert ("expanded-row-content" in content) == table_preview
    assert ("table-statistic" in content) == table_preview


def test_datacard_json_formatting(cache, json_db):
    r"""Test Datacard.json always returns formatted JSON.

    Even when the source JSON file uses compact
    single-line formatting,
    the output should be pretty-printed
    with an indent of 2.

    """
    dataset = audbcards.Dataset(json_db.name, pytest.VERSION, cache_root=cache)
    datacard = audbcards.Datacard(dataset, cache_root=cache)

    expected = (
        ".. code:: json\n"
        "\n"
        "  [\n"
        "    {\n"
        '      "role": "human",\n'
        '      "text": "What\'s the weather?"\n'
        "    },\n"
        "    {\n"
        '      "role": "assistant",\n'
        '      "text": "Nice"\n'
        "    }\n"
        "  ]\n"
    )

    # Verify formatting with already well-formatted JSON
    result = datacard.json()
    assert result == expected

    # Overwrite cached JSON file with compact single-line format
    json_file = audb.load_media(
        dataset.name,
        dataset.example_json,
        version=dataset.version,
        verbose=False,
    )[0]
    with open(json_file, encoding="utf-8") as fp:
        data = json.load(fp)
    with open(json_file, "w", encoding="utf-8") as fp:
        json.dump(data, fp, separators=(",", ":"))

    # Verify output is still properly formatted
    result = datacard.json()
    assert result == expected


def test_datacard_schemes_table_quoting():
    r"""Schemes with labels containing double quotes render valid CSV.

    The ``Schemes`` section of a data card is rendered
    as an RST ``csv-table`` directive,
    where every cell is wrapped in double quotes.
    Double quotes inside a label
    (as in ``saarbruecken-voice-database-v2``)
    must be CSV-escaped by doubling them,
    otherwise the directive fails to parse
    with ``',' expected after '"'``.

    """
    template_dir = os.path.join(
        os.path.dirname(audbcards.core.datacard.__file__),
        "templates",
    )
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
    )
    template = environment.get_template("datacard_schemes.j2")

    schemes_table = [
        ["ID", "Dtype", "Labels"],
        ["diagnosis", "str", 'foo, "quoted" label, bar'],
        ["gender", "str", "female, male"],
    ]
    content = template.render(schemes_table=schemes_table)

    # Embedded double quotes are escaped by doubling them
    assert '""quoted""' in content

    # Every data row parses back to the original cells,
    # mirroring how docutils parses the csv-table directive
    rows = [
        line.strip() for line in content.splitlines() if line.strip().startswith('"')
    ]
    assert len(rows) == len(schemes_table)
    for line, expected_row in zip(rows, schemes_table):
        parsed = next(csv.reader([line], skipinitialspace=True))
        assert parsed == [str(cell) for cell in expected_row]


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
    datacard = audbcards.Datacard(dataset, path=datacard_path, cache_root=cache)

    # Without specifying sphinx src and build dirs,
    # we do not expect a PNG file
    distribution_str = datacard.file_duration_distribution
    build_dir = audeer.mkdir(tmpdir, "build", "html")
    src_dir = audeer.mkdir(tmpdir, "docs")
    image_file = audeer.path(
        build_dir,
        datacard.path,
        db.name,
        f"{db.name}-{pytest.VERSION}-file-duration-distribution.png",
    )
    assert not os.path.exists(image_file)
    if expected_min == expected_max:
        expected_distribution_str = f"each file is {expected_max:.1f} s"
    else:
        expected_distribution_str = f"{expected_min:.1f} s .. {expected_max:.1f} s"
    assert expected_distribution_str == distribution_str

    # Set sphinx src and build dir and execute again
    datacard.sphinx_build_dir = build_dir
    datacard.sphinx_src_dir = src_dir
    distribution_str = datacard.file_duration_distribution
    if expected_min != expected_max:
        assert os.path.exists(image_file)
        expected_distribution_str = (
            f"{expected_min:.1f} s "
            f"|{db.name}-{pytest.VERSION}-file-duration-distribution| "
            f"{expected_max:.1f} s"
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
    datacard = audbcards.Datacard(dataset, path=datacard_path, cache_root=cache)

    # Execute player
    # without specifying sphinx src and build dirs
    expected_player_str = ""
    player_str = datacard.player()
    build_dir = audeer.mkdir(tmpdir, "build", "html")
    src_dir = audeer.mkdir(tmpdir, "docs")
    media_file = audeer.path(
        build_dir,
        datacard.path,
        db.name,
        datacard.dataset.example_media,
    )
    image_file = audeer.path(
        src_dir,
        datacard.path,
        db.name,
        f"{db.name}-{pytest.VERSION}-player-waveform.png",
    )
    assert not os.path.exists(media_file)
    assert not os.path.exists(image_file)
    assert player_str == expected_player_str

    # With sphinx source dir
    expected_player_str = (
        f".. image:: ./{db.name}/{db.name}-{pytest.VERSION}-player-waveform.png\n\n"
    )
    datacard.sphinx_src_dir = src_dir
    player_str = datacard.player()
    assert not os.path.exists(media_file)
    assert os.path.exists(image_file)
    assert player_str == expected_player_str
    os.remove(image_file)

    # With sphinx build dir
    expected_player_str = (
        ".. raw:: html\n"
        "\n"
        f'    <p><audio controls src="./{db.name}/{datacard.dataset.example_media}">'
        f"</audio></p>"
    )
    datacard.sphinx_src_dir = None
    datacard.sphinx_build_dir = build_dir
    player_str = datacard.player()
    assert os.path.exists(media_file)
    assert not os.path.exists(image_file)
    assert player_str == expected_player_str
    os.remove(media_file)

    # With sphinx source dir and build dir
    expected_player_str = (
        f".. image:: ./{db.name}/{db.name}-{pytest.VERSION}-player-waveform.png\n"
        "\n"
        ".. raw:: html\n"
        "\n"
        f'    <p><audio controls src="./{db.name}/{datacard.dataset.example_media}">'
        f"</audio></p>"
    )
    datacard.sphinx_build_dir = build_dir
    datacard.sphinx_src_dir = src_dir
    player_str = datacard.player()
    assert os.path.exists(media_file)
    assert os.path.exists(image_file)
    assert expected_player_str == player_str

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
