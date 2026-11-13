import os
import types

import pytest

import audeer

import audbcards.sphinx


@pytest.mark.parametrize(
    "sections",
    [
        # folder/name, header, repositories, example
        [("data-a", "A", [], True)],
        # folder/name, header, repositories, example, table preview
        [("data-a", "A", [], True, False)],
        # mixed entries
        [("data-a", "A", [], True), ("data-b", "B", [], False, False)],
    ],
)
def test_builder_finished(tmpdir, sections):
    r"""Test cleanup of generated files for all section formats."""
    src_dir = audeer.mkdir(tmpdir, "docs")
    files = []
    for path, *_ in sections:
        audeer.mkdir(src_dir, path)
        audeer.touch(src_dir, path, "db.rst")
        files.append(audeer.touch(src_dir, f"{path}.rst"))
        files.append(audeer.touch(src_dir, f"{path}.csv"))

    app = types.SimpleNamespace(
        srcdir=src_dir,
        config=types.SimpleNamespace(audbcards_datasets=sections),
    )
    audbcards.sphinx.builder_finished(app, None)

    for path, *_ in sections:
        assert not os.path.exists(os.path.join(src_dir, path))
    for file in files:
        assert not os.path.exists(file)
