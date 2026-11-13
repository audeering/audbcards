import contextlib
import os
import types

import pandas as pd
import pytest

import audb
import audeer

import audbcards.sphinx


REPOSITORY = audb.Repository("repo", "host", "file-system")


def sphinx_app(src_dir, build_dir, sections):
    r"""Minimal stand-in for a sphinx application."""
    return types.SimpleNamespace(
        srcdir=src_dir,
        builder=types.SimpleNamespace(outdir=build_dir),
        config=types.SimpleNamespace(
            audbcards_datasets=sections,
            audbcards_templates=None,
        ),
        add_css_file=lambda file: None,
        add_js_file=lambda file: None,
    )


@pytest.mark.parametrize(
    "sections",
    [
        [{"folder": "data-a"}],
        [{"folder": "data-a"}, {"folder": "data-b", "table_preview": False}],
        # deprecated tuple entries
        [("data-a", "A", [], True)],
        [("data-a", "A", [], True, False)],
        # mixed entries
        [("data-a", "A", [], True), {"folder": "data-b"}],
    ],
)
def test_builder_finished(tmpdir, sections):
    r"""Test cleanup of generated files for all section formats."""
    src_dir = audeer.mkdir(tmpdir, "docs")
    folders = [audbcards.sphinx.section_folder(section) for section in sections]
    files = []
    for folder in folders:
        audeer.mkdir(src_dir, folder)
        audeer.touch(src_dir, folder, "db.rst")
        files.append(audeer.touch(src_dir, f"{folder}.rst"))
        files.append(audeer.touch(src_dir, f"{folder}.csv"))

    app = sphinx_app(src_dir, None, sections)
    audbcards.sphinx.builder_finished(app, None)

    for folder in folders:
        assert not os.path.exists(os.path.join(src_dir, folder))
    for file in files:
        assert not os.path.exists(file)


def test_builder_finished_invalid_sections(tmpdir):
    r"""Test cleanup does not fail for invalid entries.

    Cleanup also runs after an error,
    e.g. raised for an invalid entry,
    and should not raise another error.

    Folders outside of the sphinx source folder,
    or the source folder itself,
    must not be deleted.

    """
    src_dir = audeer.mkdir(tmpdir, "docs")
    folder = audeer.mkdir(src_dir, "data-a")
    src_file = audeer.touch(src_dir, "index.rst")
    sibling_file = audeer.touch(tmpdir, "other.rst")
    sections = [
        {"header": "No folder"},
        (),
        "invalid",
        {"folder": None},
        {"folder": 1},
        {"folder": ""},
        {"folder": "."},
        {"folder": ".."},
        {"folder": "data-a/../.."},
        (str(tmpdir), "Header", [], True),
        {"folder": "data-a"},
    ]
    app = sphinx_app(src_dir, None, sections)
    audbcards.sphinx.builder_finished(app, None)
    assert not os.path.exists(folder)
    assert os.path.exists(src_file)
    assert os.path.exists(sibling_file)


@pytest.mark.parametrize(
    "sections, expected",
    [
        (
            [{"folder": "data"}],
            [("data", "data", None, True, True)],
        ),
        (
            [
                {
                    "folder": "data-a",
                    "header": "Data A",
                    "repositories": REPOSITORY,
                    "example": False,
                    "table_preview": False,
                },
                {"folder": "data-b", "table_preview": False},
            ],
            [
                ("data-a", "Data A", [REPOSITORY], False, False),
                ("data-b", "data-b", None, True, False),
            ],
        ),
        # deprecated tuple entries
        (
            [
                ("data-a", "Data A", REPOSITORY, False),
                ("data-b", "Data B", [REPOSITORY], True, False),
            ],
            [
                ("data-a", "Data A", [REPOSITORY], False, True),
                ("data-b", "Data B", [REPOSITORY], True, False),
            ],
        ),
    ],
)
def test_builder_inited(tmpdir, monkeypatch, sections, expected):
    r"""Test config entries are passed on to data cards.

    ``None`` as expected repositories
    stands for the repositories
    from :attr:`audb.config.REPOSITORIES`.

    """
    src_dir = audeer.mkdir(tmpdir, "docs")
    build_dir = audeer.mkdir(tmpdir, "build")
    app = sphinx_app(src_dir, build_dir, sections)
    current_repos = audb.config.REPOSITORIES

    calls = []

    def available(*args, **kwargs):
        calls.append({"repositories": audb.config.REPOSITORIES})
        return pd.DataFrame({"version": ["1.0.0"]}, index=["db"])

    class Datacard:
        def __init__(self, dataset, *, path, example, table_preview, **kwargs):
            calls[-1].update(
                {"folder": path, "example": example, "table_preview": table_preview}
            )

        def save(self, file):
            pass

    def create_datasets_page(datasets, rst_file, *, datacards_path, header):
        calls[-1]["header"] = header

    monkeypatch.setattr(audb, "available", available)
    monkeypatch.setattr(
        audbcards.sphinx,
        "Dataset",
        lambda name, version: types.SimpleNamespace(name=name),
    )
    monkeypatch.setattr(audbcards.sphinx, "Datacard", Datacard)
    monkeypatch.setattr(
        audbcards.sphinx,
        "create_datasets_page",
        create_datasets_page,
    )

    if any(isinstance(section, tuple) for section in sections):
        context = pytest.warns(FutureWarning, match="Use a dictionary instead")
    else:
        context = contextlib.nullcontext()
    with context:
        audbcards.sphinx.builder_inited(app)

    assert len(calls) == len(expected)
    for call, (folder, header, repositories, example, table_preview) in zip(
        calls, expected
    ):
        if repositories is None:
            repositories = current_repos
        assert call == {
            "repositories": repositories,
            "folder": folder,
            "header": header,
            "example": example,
            "table_preview": table_preview,
        }
    assert audb.config.REPOSITORIES is current_repos


def test_builder_inited_restores_repositories(tmpdir, monkeypatch):
    r"""Test repositories are restored if building data cards fails."""
    src_dir = audeer.mkdir(tmpdir, "docs")
    build_dir = audeer.mkdir(tmpdir, "build")
    sections = [{"folder": "data", "repositories": REPOSITORY}]
    app = sphinx_app(src_dir, build_dir, sections)

    def available(*args, **kwargs):
        assert audb.config.REPOSITORIES == [REPOSITORY]
        raise RuntimeError("failed")

    monkeypatch.setattr(audb, "available", available)
    current_repos = audb.config.REPOSITORIES

    with pytest.raises(RuntimeError, match="failed"):
        audbcards.sphinx.builder_inited(app)

    assert audb.config.REPOSITORIES is current_repos


def test_builder_inited_invalid_folder(tmpdir, monkeypatch):
    r"""Test invalid folders raise before any data card is deleted."""
    src_dir = audeer.mkdir(tmpdir, "docs")
    build_dir = audeer.mkdir(tmpdir, "build")
    datacard_file = audeer.touch(audeer.mkdir(src_dir, "data-a"), "db.rst")
    sections = [{"folder": "data-a"}, {"folder": ".."}]
    app = sphinx_app(src_dir, build_dir, sections)

    def available(*args, **kwargs):
        raise AssertionError("audb.available() should not be called")

    monkeypatch.setattr(audb, "available", available)

    with pytest.raises(ValueError, match="needs to be a sub folder"):
        audbcards.sphinx.builder_inited(app)
    assert os.path.exists(datacard_file)


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("data", "data"),
        ("data/", "data"),
        ("data/sub", os.path.join("data", "sub")),
        ("data/../other", "other"),
    ],
)
def test_datacard_folder(tmpdir, folder, expected):
    r"""Test data card folder inside sphinx source folder."""
    src_dir = audeer.mkdir(tmpdir, "docs")
    path = audbcards.sphinx.datacard_folder(src_dir, folder)
    assert path == os.path.join(audeer.path(src_dir), expected)


@pytest.mark.parametrize(
    "folder, error, match",
    [
        (None, TypeError, "needs to be a string"),
        (1, TypeError, "needs to be a string"),
        ("", ValueError, "needs to be a sub folder"),
        (".", ValueError, "needs to be a sub folder"),
        ("data/..", ValueError, "needs to be a sub folder"),
        ("..", ValueError, "needs to be a sub folder"),
        ("../docs-other", ValueError, "needs to be a sub folder"),
        (os.path.abspath(os.sep), ValueError, "needs to be a sub folder"),
    ],
)
def test_datacard_folder_errors(tmpdir, folder, error, match):
    r"""Test errors for folders outside sphinx source folder."""
    src_dir = audeer.mkdir(tmpdir, "docs")
    with pytest.raises(error, match=match):
        audbcards.sphinx.datacard_folder(src_dir, folder)


def test_parse_section():
    r"""Test parsing of config entries and their defaults."""
    current_repos = audb.config.REPOSITORIES
    assert audbcards.sphinx.parse_section({"folder": "data"}) == {
        "folder": "data",
        "header": "data",
        "repositories": current_repos,
        "example": True,
        "table_preview": True,
    }
    section = {
        "folder": "data",
        "header": "Data",
        "repositories": REPOSITORY,
        "example": False,
        "table_preview": False,
    }
    assert audbcards.sphinx.parse_section(section) == section


@pytest.mark.parametrize(
    "section, expected",
    [
        (
            ("data", "Data", REPOSITORY, False),
            {
                "folder": "data",
                "header": "Data",
                "repositories": REPOSITORY,
                "example": False,
                "table_preview": True,
            },
        ),
        (
            ["data", "Data", REPOSITORY, False, False],
            {
                "folder": "data",
                "header": "Data",
                "repositories": REPOSITORY,
                "example": False,
                "table_preview": False,
            },
        ),
    ],
)
def test_parse_section_tuple(section, expected):
    r"""Test deprecated tuple entries."""
    with pytest.warns(FutureWarning, match="Use a dictionary instead"):
        assert audbcards.sphinx.parse_section(section) == expected


@pytest.mark.parametrize(
    "section, error, match",
    [
        ({}, ValueError, "needs to have the key 'folder'"),
        ({"header": "Data"}, ValueError, "needs to have the key 'folder'"),
        (
            {"folder": "data", "table_previews": False},
            ValueError,
            r"Unknown key\(s\) \['table_previews'\]",
        ),
        (("data", "Data", REPOSITORY), ValueError, "four or five values"),
        (
            ("data", "Data", REPOSITORY, True, True, True),
            ValueError,
            "four or five values",
        ),
        ("data", TypeError, "needs to be a dictionary"),
    ],
)
def test_parse_section_errors(section, error, match):
    r"""Test errors for invalid config entries."""
    with pytest.raises(error, match=match):
        audbcards.sphinx.parse_section(section)
