import io
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import audbackend
import audeer
import audformat

import audbcards
from audbcards.core.utils import _RangeFile


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


class FakeMinioClient:
    r"""Fake MinIO client serving a single file from memory."""

    def __init__(self, data: bytes):
        self.data = data
        self.requests = []

    def stat_object(self, bucket, path):
        r"""Return object holding file size."""

        class Stat:
            size = len(self.data)

        return Stat()

    def get_object(self, bucket, path, *, offset, length):
        r"""Return response holding requested range of file."""
        self.requests.append((offset, length))
        data = self.data[offset : offset + length]

        class Response(io.BytesIO):
            def release_conn(self):
                pass

        return Response(data)


def test_range_file():
    r"""Test random access to file with range requests."""
    data = bytes(range(256)) * 4
    client = FakeMinioClient(data)
    f = _RangeFile(client, "bucket", "path")
    assert f.readable()
    assert f.seekable()
    assert f.tell() == 0
    assert f.read(10) == data[:10]
    assert f.tell() == 10
    assert f.seek(5, io.SEEK_CUR) == 15
    assert f.read(3) == data[15:18]
    assert f.seek(-4, io.SEEK_END) == len(data) - 4
    assert f.read() == data[-4:]
    assert f.read(1) == b""
    assert f.seek(len(data) + 10) == len(data) + 10
    assert f.read(1) == b""
    with pytest.raises(ValueError):
        f.seek(-1)
    with pytest.raises(ValueError):
        f.seek(0, 3)


@pytest.mark.parametrize("rows", [0, 3, 200_000])
def test_parquet_preview_range_file(tmpdir, rows):
    r"""Test reading PARQUET preview with range requests."""
    table = pa.table(
        {
            "file": [f"f{n}.wav" for n in range(rows)],
            # Random strings to avoid compression
            "text": [os.urandom(16).hex() for _ in range(rows)],
        }
    )
    path = audeer.path(tmpdir, "table.parquet")
    pq.write_table(table, path, row_group_size=50_000)
    with open(path, "rb") as fp:
        data = fp.read()
    client = FakeMinioClient(data)
    num_rows, preview = audbcards.core.utils.parquet_preview(
        _RangeFile(client, "bucket", "path"),
        5,
    )
    assert num_rows == rows
    assert preview.equals(table.slice(0, 5))
    # Only part of the file is requested
    if rows == 200_000:
        assert sum(length for _, length in client.requests) < len(data) / 2


def test_open_backend_file(tmpdir):
    r"""Test opening file on backend."""
    host = audeer.mkdir(tmpdir, "host")
    audbackend.backend.FileSystem.create(host, "repo")
    backend = audbackend.backend.FileSystem(host, "repo")
    src_file = audeer.path(tmpdir, "file.txt")
    with open(src_file, "w") as fp:
        fp.write("content")
    with backend:
        interface = audbackend.interface.Versioned(backend)
        interface.put_file(src_file, "/a/file.txt", "1.0.0")
        with audbcards.core.utils.open_backend_file(
            interface,
            "/a/file.txt",
            "1.0.0",
        ) as fp:
            assert fp.read() == b"content"
        # Unsupported interface
        interface = audbackend.interface.Unversioned(backend)
        assert (
            audbcards.core.utils.open_backend_file(interface, "/a/file.txt", "1.0.0")
            is None
        )
