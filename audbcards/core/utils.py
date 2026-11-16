from collections.abc import Sequence
import io
import typing

import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

import audbackend
import audeer
import audformat


class _RangeFile(io.RawIOBase):
    r"""Read-only, seekable file object of a file on a MinIO/S3 backend.

    Each read is a range request,
    which allows reading parts of a file
    without downloading it completely.

    Args:
        client: MinIO client
        bucket: bucket name
        path: path of file inside bucket

    """

    def __init__(self, client, bucket: str, path: str):
        self._client = client
        self._bucket = bucket
        self._path = path
        self._size = client.stat_object(bucket, path).size
        self._pos = 0

    def readable(self) -> bool:
        r"""File is readable."""
        return True

    def seekable(self) -> bool:
        r"""File is seekable."""
        return True

    def tell(self) -> int:
        r"""Current position in file."""
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        r"""Change position in file."""
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if pos < 0:
            raise ValueError(f"Negative seek position: {pos}")
        self._pos = pos
        return self._pos

    def readinto(self, buffer) -> int:
        r"""Read bytes into buffer, using a single range request."""
        length = min(len(buffer), self._size - self._pos)
        if length <= 0:
            return 0
        response = self._client.get_object(
            self._bucket,
            self._path,
            offset=self._pos,
            length=length,
        )
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


def open_backend_file(
    interface: audbackend.interface.Base,
    path: str,
    version: str,
) -> typing.BinaryIO | None:
    r"""Open file on backend for random access reading.

    Supports versioned interfaces
    of the MinIO/S3 and file-system backends,
    which allow to read parts of a file
    without downloading it.

    This relies on private attributes of :mod:`audbackend`,
    and should be replaced,
    when :mod:`audbackend` provides a public API
    (https://github.com/audeering/audbackend/issues/317).
    The backend connection needs to be open
    while reading the file.

    Args:
        interface: backend interface
        path: path of file on backend
        version: version of file

    Returns:
        binary file object,
        or ``None`` if backend or interface are not supported

    """
    if not isinstance(interface, audbackend.interface.Versioned):
        return None
    backend = interface.backend
    path = interface._path_with_version(path, version)
    minio = getattr(audbackend.backend, "Minio", None)
    if minio is not None and isinstance(backend, minio):
        return _RangeFile(backend._client, backend.repository, backend.path(path))
    if isinstance(backend, audbackend.backend.FileSystem):
        return open(backend._expand(path), "rb")
    return None


def parquet_preview(
    file: str | typing.BinaryIO,
    preview_rows: int,
) -> tuple[int, pa.Table]:
    r"""Number of rows and first rows of a PARQUET file.

    Reads only the footer
    and the first pages of the first row group,
    without loading the complete file into memory.

    Args:
        file: path to PARQUET file or binary file object
        preview_rows: maximum number of rows of the preview

    Returns:
        number of rows of the file,
        and table holding the first ``preview_rows`` rows

    """
    # Small buffer and no pre-buffering,
    # to read only the first pages of each column
    parquet_file = pq.ParquetFile(
        file,
        buffer_size=1024 * 1024,
        pre_buffer=False,
    )
    rows = parquet_file.metadata.num_rows
    schema = parquet_file.schema_arrow
    batch = next(parquet_file.iter_batches(batch_size=preview_rows), None)
    if batch is None:
        preview = schema.empty_table()
    else:
        preview = pa.Table.from_batches([batch], schema=schema)
    return rows, preview


def format_schemes(
    schemes: dict[str, audformat.Scheme],
    excludes: Sequence[str] = ["duration"],
    max_schemes: int = 15,
) -> str:
    """Convert schemes object into string.

    It lists the main annotation schemes
    of the datasets,
    and collects additional information
    on schemes named ``'emotion'`` and ``'speaker'``,
    e.g. ``'speaker: [age, gender, language]'``.

    """
    # Filter schemes
    filtered_schemes = []
    emotion = []
    speaker = []
    for scheme in schemes:
        if scheme in excludes:
            continue
        # schemes[scheme] entries are always dictionaries,
        # so we don't have to check for that
        if scheme == "emotion":
            try:
                labels = schemes[scheme]._labels_to_list()
                labels = audeer.flatten_list(labels)
                emotion = [{scheme: labels}]
                max_schemes -= 1
            except KeyError:
                emotion = [scheme]
        elif scheme == "speaker":
            try:
                labels = schemes[scheme]._labels_to_dict()
                labels = list(labels.values())
                # Store the dictionary keys for speaker
                # as those are gender, language, ...
                # Keys are the same for all entries,
                # using the first one is enough
                labels = list(labels[0].keys())
                if labels:
                    speaker = [{scheme: labels}]
                else:
                    speaker = [scheme]
                max_schemes -= 1
            except (KeyError, AttributeError):
                speaker = [scheme]
        else:
            filtered_schemes.append(scheme)
    # Force emotion and speaker to the beginning of the list
    filtered_schemes = emotion + speaker + filtered_schemes
    # Limit to maximum number of schemes and add '...' for longer once
    max_schemes = max(max_schemes, 2)
    filtered_schemes = limit_presented_samples(filtered_schemes, max_schemes)
    # Format the information for display
    info_str = ""
    for scheme in filtered_schemes:
        if isinstance(scheme, dict):
            key = list(scheme.keys())[0]
            info_str += f"{key}: ["
            for label in scheme[key]:
                info_str += f"{label}, "
            info_str = info_str[:-2] + "], "
        else:
            info_str += f"{scheme}, "
    info_str = info_str[:-2]

    return info_str


def limit_presented_samples(
    samples: Sequence,
    limit: int,
    replacement_text: str = "...",
) -> list:
    r"""Limit the printing of sequences.

    If the sequence contains too many samples,
    they will be cut out in the center.

    Args:
        samples: sequence of samples to list on screen
        limit: maximum number to present
        replacement_text: text shown instead of removed samples

    Returns:
        string listing the samples

    """
    if len(samples) >= limit:
        samples = samples[: limit // 2] + [replacement_text] + samples[-limit // 2 :]
    return samples


def set_plot_margins(
    *,
    left=0,
    bottom=0,
    right=1,
    top=1,
    wspace=0,
    hspace=0,
):
    r"""Set the margins in a plot.

    As default it will remove all margins.
    For details on arguments,
    see :func:`matplotlib.pyplot.subplots_adjust`.

    """
    plt.subplots_adjust(
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
