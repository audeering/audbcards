import io
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import audbackend
import audeer
import audformat

import audbcards


@pytest.mark.parametrize(
    "scheme_names, scheme_dtypes, labels, expected",
    [
        (
            ["emotion", "age", "gender", "language", "speaker"],
            ["str", "int", "str", "str", "int"],
            [
                ["happy", "sad"],
                None,
                ["female", "male"],
                ["DE", "EN"],
                "speaker",
            ],
            "emotion: [happy, sad], speaker: [age, gender, language], "
            "age, gender, language",
        ),
        (
            ["emotion", "age", "gender", "speaker", "audio_quality"],
            ["str", "int", "str", "int", "str"],
            [["happy", "sad"], None, ["female", "male"], "speaker", ["good", "bad"]],
            "emotion: [happy, sad], speaker: [age, gender], "
            "age, audio_quality, gender",
        ),
    ],
)
def test_format_schemes(scheme_names, scheme_dtypes, labels, expected):
    # Init database to contain schemes
    db = audformat.Database(name="db")
    for i, scheme_name in enumerate(scheme_names):
        # Create actual schemes
        if scheme_name == "speaker":
            db["speaker"] = audformat.MiscTable(
                pd.Index([0], dtype="Int8", name="speaker")
            )
            if "age" in scheme_names:
                db["speaker"]["age"] = audformat.Column(scheme_id="age")
            if "gender" in scheme_names:
                db["speaker"]["gender"] = audformat.Column(scheme_id="gender")
            if "language" in scheme_names:
                db["speaker"]["language"] = audformat.Column(scheme_id="language")
        db.schemes[scheme_name] = audformat.Scheme(
            dtype=scheme_dtypes[i],
            labels=labels[i],
        )
    # Generate scheme str with format_scheme()
    scheme_str = audbcards.core.utils.format_schemes(db.schemes)
    assert scheme_str == expected


@pytest.mark.parametrize(
    "values, bins, expected_min, expected_max, expected_edges, expected_counts",
    [
        ([], 4, 0, 0, [], []),
        ([np.nan], 4, 0, 0, [], []),
        ([2.0, 2.0, 2.0], 4, 2, 2, [2, 2], [3]),
        ([0.0, 1.0, 1.5, 4.0], 4, 0, 4, [0, 1, 2, 3, 4], [1, 2, 0, 1]),
        ([0.0, np.nan, 4.0], 2, 0, 4, [0, 2, 4], [1, 1]),
        (np.array([1.0, 3.0]), 2, 1, 3, [1, 2, 3], [1, 1]),
    ],
)
def test_histogram(
    values,
    bins,
    expected_min,
    expected_max,
    expected_edges,
    expected_counts,
):
    histogram = audbcards.core.utils.histogram(values, bins=bins)
    assert histogram["min"] == expected_min
    assert histogram["max"] == expected_max
    np.testing.assert_equal(histogram["bin_edges"], expected_edges)
    np.testing.assert_equal(histogram["counts"], expected_counts)
    assert histogram["counts"].dtype == np.int64


@pytest.mark.parametrize("sample", [["a", "b", "c", "d", "e"]])
@pytest.mark.parametrize(
    "limit, replacement_text, expected",
    [
        (2, "...", ["a", "...", "e"]),
        (2, "###", ["a", "###", "e"]),
        (4, "...", ["a", "b", "...", "d", "e"]),
    ],
)
def test_limit_presented_samples(sample, limit, replacement_text, expected):
    limited_sample = audbcards.core.utils.limit_presented_samples(
        sample, limit, replacement_text
    )
    assert limited_sample == expected


class CountingFile(io.RawIOBase):
    r"""Raw file object counting the number of read bytes."""

    def __init__(self, raw):
        self.raw = raw
        self.read_bytes = 0

    def readable(self):
        r"""File is readable."""
        return True

    def seekable(self):
        r"""File is seekable."""
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        r"""Change position in file."""
        return self.raw.seek(offset, whence)

    def tell(self):
        r"""Current position in file."""
        return self.raw.tell()

    def readinto(self, buffer):
        r"""Read bytes into buffer and count them."""
        n = self.raw.readinto(buffer)
        self.read_bytes += n
        return n

    def close(self):
        r"""Close file."""
        self.raw.close()
        super().close()


@pytest.mark.parametrize("rows", [0, 3, 200_000])
def test_parquet_preview_backend_file(tmpdir, monkeypatch, rows):
    r"""Test reading PARQUET preview from file opened on backend."""
    table = pa.table(
        {
            "file": [f"f{n}.wav" for n in range(rows)],
            # Random strings to avoid compression
            "text": [os.urandom(16).hex() for _ in range(rows)],
        }
    )
    path = audeer.path(tmpdir, "table.parquet")
    pq.write_table(table, path, row_group_size=50_000)

    # Track bytes read from backend
    files = []
    open_file = audbackend.backend.FileSystem._open_file

    def counting_open_file(self, path):
        files.append(CountingFile(open_file(self, path)))
        return files[-1]

    monkeypatch.setattr(
        audbackend.backend.FileSystem,
        "_open_file",
        counting_open_file,
    )

    host = audeer.mkdir(tmpdir, "host")
    audbackend.backend.FileSystem.create(host, "repo")
    backend = audbackend.backend.FileSystem(host, "repo")
    with backend:
        interface = audbackend.interface.Versioned(backend)
        interface.put_file(path, "/table.parquet", "1.0.0")
        with interface.open_file("/table.parquet", "1.0.0") as fp:
            num_rows, preview = audbcards.core.utils.parquet_preview(fp, 5)
    assert num_rows == rows
    assert preview.equals(table.slice(0, 5))
    # Only part of the file is read
    if rows == 200_000:
        assert files[0].read_bytes < os.path.getsize(path) / 2
