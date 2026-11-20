from collections.abc import Sequence
import contextlib
import functools
import inspect
import os
import pickle
import random
import re
import tempfile
import typing

import jinja2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import audb
import audbackend
import audeer
import audformat

from audbcards.core import utils
from audbcards.core.config import config


rng = random.Random(123)


class _Dataset:
    _table_related_cached_properties = [
        "segment_duration_histogram",
        "segments",
        "tables_columns",
        "tables_preview",
        "tables_rows",
    ]
    """Cached properties relying on table data.

    Most of the cached properties
    rely on the dependency table,
    the header of a dataset,
    and misc tables used as scheme labels.
    Some might also need to load filewise or segmented tables,
    to gather more information.

    Persistence of table related cached properties
    depends on the ``load_tables`` argument
    of :class:`audbcards.Dataset`.
    If ``load_tables`` is ``True``,
    :meth:`audbcards.Dataset._cached_properties`
    is asked to cache table related cached properties as well.
    If ``load_tables`` is ``False``,
    :meth:`audbcards.Dataset._cached_properties`
    is asked to exclude all cached properties,
    listed in ``_table_related_cached_properties``.
    Which means,
    ``_table_related_cached_properties`` has to list all cached properties,
    that will load filewise or segmented tables.

    If a dataset exists in cache,
    but does not store table related cached properties,
    a call to :class:`audbcards.Dataset`
    with ``load_tables`` is ``True``,
    will update the cache.

    """

    @classmethod
    def create(
        cls,
        name: str,
        version: str,
        *,
        cache_root: str | None = None,
        load_tables: bool = True,
    ):
        r"""Instantiate Dataset Object."""
        if cache_root is None:
            cache_root = os.environ.get("AUDBCARDS_CACHE_ROOT") or config.CACHE_ROOT
        dataset_cache_filename = cls._dataset_cache_path(name, version, cache_root)

        if os.path.exists(dataset_cache_filename):
            obj = cls._load_pickled(dataset_cache_filename)
            # Cache files written by older versions of audbcards
            # might contain private attributes,
            # like the header or all segments of the dataset,
            # which are removed by storing the cache again
            cache_again = obj.__dict__.keys() != obj.__getstate__().keys()
            # Remove those attributes from the loaded object as well,
            # including obsolete properties of older versions,
            # e.g. `file_durations`,
            # which might require a lot of memory
            obj.__dict__ = obj.__getstate__()
            # Load cached properties,
            # that require to load filewise or segmented tables,
            # if they haven't been cached before.
            if load_tables:
                for cached_property in cls._table_related_cached_properties:
                    # Check if property has been cached,
                    # see https://stackoverflow.com/a/59740750
                    if cached_property not in obj.__dict__:
                        cache_again = True
                        # Request property to fill their cached value
                        getattr(obj, cached_property)
            if cache_again:
                cls._save_pickled(obj, dataset_cache_filename)
            # Cache root is not stored in the cache file,
            # as the cache folder might have been moved.
            # It is set after checking for `cache_again`,
            # as `__getstate__()` excludes it
            obj.cache_root = audeer.mkdir(cache_root)
            obj._release_memory()
            return obj

        obj = cls(name, version, cache_root, load_tables)
        # Visit cached properties to fill their cache values
        if load_tables:
            exclude = []
        else:
            exclude = cls._table_related_cached_properties
        obj._cached_properties(exclude=exclude)

        cls._save_pickled(obj, dataset_cache_filename)
        obj._release_memory()
        return obj

    def __init__(
        self,
        name: str,
        version: str,
        cache_root: str | None = None,
        load_tables: bool = True,
    ):
        self.cache_root = audeer.mkdir(cache_root)
        r"""Cache root folder."""

        # Store name and version in private attributes here,
        # ``self.name`` and ``self.version``
        # are implemented as cached properties below
        self._name = name
        self._version = version

        # Private attributes,
        # used inside corresponding properties.
        self._header = self._load_header()
        self._deps = self._load_dependencies()
        self._repository_object = self._load_repository_object()  # load before backend
        self._backend = self._load_backend()

        # Clean up cache
        # by removing all other versions of the same dataset
        # to reduce its storage size in CI runners
        versions = audeer.list_dir_names(
            audeer.path(cache_root, name),
            basenames=True,
        )
        other_versions = [v for v in versions if v != version]
        for other_version in other_versions:
            audeer.rmdir(cache_root, name, other_version)

    def __getstate__(self) -> dict[str, object]:
        r"""Select attributes stored in the pickled cache file.

        Only public cached properties,
        that have already been computed,
        are stored.
        Private attributes,
        like the header,
        the dependency table,
        or the backend,
        are not stored,
        as they might require a lot of memory.

        Returns:
            dictionary with property name and value

        """
        public_cached_properties = self._cached_property_names(private=False)
        return {
            name: value
            for name, value in self.__dict__.items()
            if name in public_cached_properties
        }

    @classmethod
    def _cached_property_names(cls, *, private: bool) -> list[str]:
        r"""Names of public or private cached properties.

        Args:
            private: if ``True``,
                return names of private cached properties,
                otherwise of public ones

        Returns:
            names of cached properties

        """
        return [
            name
            for name, value in cls.__dict__.items()
            if isinstance(value, functools.cached_property)
            and name.startswith("_") == private
        ]

    def _release_memory(self, cached_properties: Sequence[str] = ()):
        r"""Release memory of attributes not needed after caching.

        Removes header,
        dependency table,
        backend,
        and all private cached properties
        from the object.
        They are loaded or computed again
        when accessed.
        The repository object is kept,
        as it requires only little memory,
        and ensures the backend is loaded again
        from the same repository,
        even if :attr:`audb.config.REPOSITORIES`
        has changed in the meantime.

        Args:
            cached_properties: public cached properties
                that should be removed as well,
                e.g. ``"tables_preview"``,
                that are not needed anymore

        """
        attributes = [
            "_backend",
            "_deps",
            "_header",
        ]
        attributes += self._cached_property_names(private=True)
        attributes += list(cached_properties)
        for attribute in attributes:
            self.__dict__.pop(attribute, None)

    @staticmethod
    def _dataset_cache_path(name: str, version: str, cache_root: str) -> str:
        r"""Generate the name of the cache file."""
        cache_dir = audeer.mkdir(cache_root, name, version)

        cache_filename = audeer.path(
            cache_dir,
            f"{name}-{version}.pkl",
        )
        return cache_filename

    @staticmethod
    def _load_pickled(path: str):
        r"""Load pickled object instance."""
        if not os.path.exists(path):
            raise FileNotFoundError()

        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _parse_text(text: str) -> str:
        """Remove unsupported characters and restrict length.

        The text is stripped from HTML tags or newlines,
        and limited to a maximum length of 100 characters.

        Args:
            text: input text

        Returns:
            parsed text

        """
        # Missing text
        if pd.isna(text):
            return ""
        # Remove newlines
        text = text.replace("\n", "\\n")
        # Remove HTML tags
        text = re.sub("<[^<]+?>", "", text)
        # Limit length
        max_characters_per_entry = 100
        if len(text) > max_characters_per_entry:
            text = text[: max_characters_per_entry - 3] + "..."
        return text

    @staticmethod
    def _save_pickled(obj, path: str):
        """Save object instance to path as pickle.

        The object is first written to a temporary file,
        which then replaces ``path`` atomically.
        This ensures ``path`` never contains a truncated pickle,
        if the process is interrupted while writing,
        or several processes write the same cache file.

        """
        audeer.mkdir(os.path.dirname(path))
        # Use unique name in same folder,
        # as `os.replace()` is only atomic on the same file system.
        # `tempfile.mkstemp()` is not used,
        # as it would restrict file permissions to the current user
        tmp_path = f"{path}.{os.getpid()}.{audeer.uid()[:8]}.tmp"
        try:
            with open(tmp_path, "xb") as f:
                pickle.dump(obj, f, protocol=4)
            os.replace(tmp_path, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)
            raise

    @property
    def backend(self) -> type[audbackend.interface.Base]:
        r"""Dataset backend object."""
        if not hasattr(self, "_backend"):  # when loaded from cache
            self._backend = self._load_backend()
        return self._backend

    @property
    def deps(self) -> audb.Dependencies:
        r"""Dataset dependency table."""
        if not hasattr(self, "_deps"):  # when loaded from cache
            self._deps = self._load_dependencies()
        return self._deps

    @property
    def header(self) -> audformat.Database:
        r"""Dataset header."""
        if not hasattr(self, "_header"):  # when loaded from cache
            self._header = self._load_header()
        return self._header

    @property
    def repository_object(self) -> audb.Repository:
        r"""Repository object containing dataset."""
        if not hasattr(self, "_repository_object"):  # when loaded from cache
            self._repository_object = self._load_repository_object()
        return self._repository_object

    @functools.cached_property
    def archives(self) -> int:
        r"""Number of archives of media files in dataset."""
        return len(set([self.deps.archive(file) for file in self.deps.media]))

    @functools.cached_property
    def author(self) -> list[str]:
        r"""Authors of the database."""
        return self.header.author

    @functools.cached_property
    def bit_depths(self) -> list[int]:
        r"""Bit depths of media files in dataset."""
        return sorted(
            list(
                set(
                    [
                        self.deps.bit_depth(file)
                        for file in self.deps.media
                        if self.deps.bit_depth(file)
                    ]
                )
            )
        )

    @functools.cached_property
    def channels(self) -> list[int]:
        r"""Channels of media files in dataset."""
        return sorted(
            list(
                set(
                    [
                        self.deps.channels(file)
                        for file in self.deps.media
                        if self.deps.channels(file)
                    ]
                )
            )
        )

    @functools.cached_property
    def description(self) -> str:
        r"""Source of the database."""
        return self.header.description

    @functools.cached_property
    def duration(self) -> pd.Timedelta:
        r"""Total duration of media files in dataset."""
        durations = [self.deps.duration(file) for file in self.deps.media]
        return pd.to_timedelta(
            sum([d for d in durations if d is not None]),
            unit="s",
        )

    @functools.cached_property
    def example_json(self) -> str | None:
        r"""Example json file.

        Path to example json file from dataset.
        The json file needs to be stored in an archive
        with at most 1000 files.
        If the json file does not meet this criterium
        or no json file is part of the dataset,
        ``None`` is returned instead.

        """
        json_idx = [n for n, ext in enumerate(self.deps().format) if ext == "json"]
        if not json_idx:
            return None
        index = rng.choice(json_idx)
        file = self.deps.files[index]
        return file if self._files_in_archive(file) <= 1000 else None

    @functools.cached_property
    def example_media(self) -> str | None:
        r"""Example media file.

        The media file is selected
        by its median duration
        from all files in the dataset
        with a duration
        between 0.5 s and 300 s.
        In addition,
        the media file needs to be stored in an archive
        with less than 100 files.
        If no media file meets this criterium,
        ``None`` is returned instead.

        """
        # Pick a meaningful duration for the example audio file
        min_dur = 0.5
        max_dur = 300  # 5 min

        media_durations = self._media_with_durations
        candidates = [
            (file, dur)
            for file, dur in media_durations
            if dur >= min_dur and dur <= max_dur
        ]

        if len(candidates) == 0:
            return None

        median_duration = np.median([dur for _, dur in candidates])

        # pick file with duration closest to median
        file, _ = min(candidates, key=lambda item: abs(item[1] - median_duration))

        return file if self._files_in_archive(file) < 100 else None

    @functools.cached_property
    def files(self) -> int:
        r"""Number of media files in dataset."""
        return len(self.deps.media)

    @functools.cached_property
    def _media_with_durations(self) -> list[tuple[str, float]]:
        r"""Media files paired with their durations.

        Non media files,
        or media files containing 0 samples
        are excluded from this list.

        """
        return [
            (file, dur) for file in self.deps.media if (dur := self.deps.duration(file))
        ]

    @functools.cached_property
    def file_duration_histogram(self) -> dict[str, float | np.ndarray]:
        r"""Minimum, maximum, and histogram of file durations in seconds.

        Non media files,
        or media files containing 0 samples
        are excluded.
        The histogram uses 1000 bins,
        spaced linearly between minimum and maximum duration.

        Returns:
            dictionary with the entries
            ``"min"``,
            ``"max"``,
            ``"bin_edges"``,
            and ``"counts"``.
            If the dataset contains no media files,
            minimum and maximum are ``0``
            and the arrays are empty

        """
        return utils.histogram(
            np.fromiter(
                (dur for _, dur in self._media_with_durations),
                dtype=np.float64,
            )
        )

    @functools.cached_property
    def formats(self) -> list[str]:
        r"""File formats of media files in dataset."""
        return sorted(list(set([self.deps.format(file) for file in self.deps.media])))

    @functools.cached_property
    def languages(self) -> list[str]:
        r"""Languages of the database."""
        return self.header.languages

    @functools.cached_property
    def iso_languages(self) -> list[str]:
        r"""Languages of the database as ISO 639-3 if possible."""
        return self._map_iso_languages(self.languages)

    @functools.cached_property
    def license(self) -> str:
        r"""License of dataset.

        If no license is given,
        ``'Unknown'`` is returned.

        """
        return self.header.license or "Unknown"

    @functools.cached_property
    def license_link(self) -> str | None:
        r"""Link to license of dataset.

        If no link is available
        ``None`` is returned.

        """
        if self.header.license_url is None or len(self.header.license_url) == 0:
            return None
        else:
            return self.header.license_url

    @functools.cached_property
    def name(self) -> str:
        r"""Name of dataset."""
        return self._name

    @functools.cached_property
    def publication_date(self) -> str:
        r"""Date dataset was uploaded to repository."""
        with self.backend.backend:
            path = self.backend.join("/", self.name, "db.yaml")
            return self.backend.date(path, self.version)

    @functools.cached_property
    def publication_owner(self) -> str:
        r"""User who uploaded dataset to repository."""
        with self.backend.backend:
            path = self.backend.join("/", self.name, "db.yaml")
            return self.backend.owner(path, self.version)

    @functools.cached_property
    def repository(self) -> str:
        r"""Repository containing the dataset."""
        return f"{self.repository_object.name}"

    @functools.cached_property
    def repository_link(self) -> str:
        r"""Link to repository in Artifactory web UI."""
        # NOTE: this needs to be changed
        # as we want to support different backends
        return (
            f"{self.repository_object.host}/"
            f"webapp/#/artifacts/browse/tree/General/"
            f"{self.repository}/"
            f"{self.name}"
        )

    @functools.cached_property
    def sampling_rates(self) -> list[int]:
        r"""Sampling rates of media files in dataset."""
        return sorted(
            list(
                set(
                    [
                        self.deps.sampling_rate(file)
                        for file in self.deps.media
                        if self.deps.sampling_rate(file)
                    ]
                )
            )
        )

    @functools.cached_property
    def schemes(self) -> list[str]:
        r"""Schemes of dataset."""
        return list(self.header.schemes)

    @functools.cached_property
    def schemes_summary(self) -> str:
        r"""Summary of dataset schemes.

        It lists all schemes in a string,
        showing additional information
        on schemes named ``'emotion'`` and ``'speaker'``,
        e.g. ``'speaker: [age, gender, language]'``.

        """
        return utils.format_schemes(self.header.schemes)

    @functools.cached_property
    def schemes_table(self) -> list[list[str]]:
        """Schemes table with name, type, min, max, labels, mappings.

        The table is represented as a dictionary
        with column names as keys.

        """
        db = self.header
        dataset_schemes = []
        for scheme_id in db.schemes:
            dataset_scheme = self._scheme_to_list(scheme_id)
            dataset_schemes.append(dataset_scheme)

        cols = self._scheme_table_columns
        data = pd.DataFrame.from_dict(dataset_schemes)[cols]
        filter = data.map(lambda d: d == [])
        data.mask(filter, other="", inplace=True)
        scheme_data = data.values.tolist()
        # Add column names
        scheme_data.insert(0, list(data))
        return scheme_data

    @functools.cached_property
    def segments(self) -> str:
        r"""Number of segments in dataset."""
        return str(self._segments.num_rows)

    @functools.cached_property
    def segment_duration_histogram(self) -> dict[str, float | np.ndarray]:
        r"""Minimum, maximum, and histogram of segment durations in seconds.

        Segments without end are excluded.
        The histogram uses 1000 bins,
        spaced linearly between minimum and maximum duration.

        Returns:
            dictionary with the entries
            ``"min"``,
            ``"max"``,
            ``"bin_edges"``,
            and ``"counts"``.
            If the dataset contains no segments,
            minimum and maximum are ``0``
            and the arrays are empty

        """
        return utils.histogram(self._segment_durations())

    def _segment_durations(self) -> np.ndarray:
        r"""Segment durations in seconds.

        Returns:
            array of segment durations,
            ``NaN`` for segments without end

        """
        if self._segments.num_rows == 0:
            return np.array([], dtype=np.float64)
        durations = pc.subtract(self._segments["end"], self._segments["start"])
        seconds = pc.divide(pc.cast(durations, pa.int64()), 1e9)
        # Segments without end result in NaN
        return seconds.to_numpy(zero_copy_only=False)

    @functools.cached_property
    def short_description(self) -> str:
        r"""Description of dataset shortened to 150 chars."""
        length = 150
        description = self.header.description or ""
        # Fix RST used signs
        description = description.replace("`", "'")
        if len(description) > length:
            description = f"{description[: length - 3]}..."
        return description

    @functools.cached_property
    def source(self) -> str:
        r"""Source of the database."""
        return self.header.source

    @functools.cached_property
    def tables(self) -> list[str]:
        """Tables of the dataset."""
        db = self.header
        tables = list(db)
        return tables

    @functools.cached_property
    def tables_columns(self) -> dict[str, int]:
        """Number of columns for each table of the dataset.

        Returns:
            dictionary with table IDs as keys
            and number of columns as values

        Examples:
            >>> ds = Dataset("emodb", "1.4.1")
            >>> ds.tables_columns["speaker"]
            3

        """
        return {table: stats["columns"] for table, stats in self._tables_stats.items()}

    @functools.cached_property
    def tables_preview(self) -> dict[str, list[list[str]]]:
        """Table preview for each table of the dataset.

        Shows the header
        and the first 5 lines for each table
        as a list of lists.
        All table values are converted to strings,
        stripped from HTML tags or newlines,
        and limited to a maximum length of 100 characters.

        Returns:
            dictionary with table IDs as keys
            and table previews as values

        Examples:
            >>> from tabulate import tabulate
            >>> ds = Dataset("emodb", "1.4.1")
            >>> preview = ds.tables_preview["speaker"]
            >>> print(tabulate(preview, headers="firstrow", tablefmt="github"))
            |   speaker |   age | gender   | language   |
            |-----------|-------|----------|------------|
            |         3 |    31 | male     | deu        |
            |         8 |    34 | female   | deu        |
            |         9 |    21 | female   | deu        |
            |        10 |    32 | male     | deu        |
            |        11 |    26 | male     | deu        |

        """
        preview = {}
        for table, stats in self._tables_stats.items():
            df = stats["preview"]
            df = df.reset_index()
            header = [df.columns.tolist()]
            # Convert timedelta columns with str(),
            # as astype("string") includes microseconds
            # for all entries since pandas 3.0
            for column in df.select_dtypes("timedelta").columns:
                df[column] = df[column].map(str, na_action="ignore")
            body = df.astype("string").values.tolist()
            # Remove unwanted chars and limit length of each entry
            body = [[self._parse_text(column) for column in row] for row in body]
            preview[table] = header + body
        return preview

    @functools.cached_property
    def tables_rows(self) -> dict[str, int]:
        """Number of rows for each table of the dataset.

        Returns:
            dictionary with table IDs as keys
            and number of rows as values

        Examples:
            >>> ds = Dataset("emodb", "1.4.1")
            >>> ds.tables_rows["speaker"]
            10

        """
        return {table: stats["rows"] for table, stats in self._tables_stats.items()}

    @functools.cached_property
    def tables_table(self) -> list[str]:
        """Tables of the dataset."""
        table_list = [["ID", "Type", "Columns"]]
        db = self.header
        for table_id in self.tables:
            table = db[table_id]
            if isinstance(table, audformat.MiscTable):
                table_type = "misc"
            else:
                table_type = table.type
            columns = ", ".join(list(table.columns))
            table_list.append([table_id, table_type, columns])

        return table_list

    @functools.cached_property
    def usage(self) -> str:
        r"""Usage of the database."""
        return self.header.usage

    @functools.cached_property
    def version(self) -> str:
        r"""Version of dataset."""
        return self._version

    def _cached_properties(
        self,
        *,
        exclude: Sequence = [],
    ) -> dict[str, object]:
        """Get list of cached properties of the object.

        When collecting the cached properties,
        it also executes their code
        in order to generate the associated values.

        Args:
            exclude: list of cached properties,
                that should not be cached

        Returns:
            dictionary with property name and value

        """
        class_items = self.__class__.__dict__.items()
        props = dict(
            (k, getattr(self, k))
            for k, v in class_items
            if (
                isinstance(v, functools.cached_property)
                and k not in exclude
                and not k.startswith("_")
            )
        )
        return props

    def _files_in_archive(self, file: str) -> int:
        """Number of files in archive for given file index.

        Args:
            file: file in dependency table

        Returns:
            number of files stored in the archive for given file

        """
        archives = self.deps().archive
        file_archive = self.deps.archive(file)
        return (archives == file_archive).sum()

    def _load_backend(self) -> type[audbackend.interface.Base]:
        r"""Load backend object containing dataset."""
        backend_interface = self.repository_object.create_backend_interface()
        return backend_interface

    def _load_dependencies(self) -> audb.Dependencies:
        r"""Load dataset dependencies."""
        return audb.dependencies(self.name, version=self.version, verbose=False)

    def _load_header(self) -> audformat.Database:
        r"""Load dataset header."""
        # Ensure misc tables are loaded
        return audb.info.header(self.name, version=self.version, load_tables=True)

    def _header_without_tables(self) -> audformat.Database:
        r"""Load header without loading misc tables."""
        return audb.info.header(self.name, version=self.version, load_tables=False)

    def _load_repository_object(self) -> audb.Repository:
        r"""Load repository object containing dataset."""
        return audb.repository(self.name, self.version)

    @functools.cached_property
    def _scheme_table_columns(self) -> list[str]:
        """Column names for the scheme table.

        Column names always include ``'ID'`` and ``'Dtype'``,
        and if defined in any scheme
        ``'Min'``,
        ``'Max'``,
        ``'Labels'``,
        ``'Mappings'``.

        """
        schemes = self.header.schemes

        if len(schemes) == 0:
            return []

        columns = ["ID", "Dtype"]

        if len(schemes) > 0:
            if any([schemes[s].minimum is not None for s in schemes]):
                columns.append("Min")
            if any([schemes[s].maximum is not None for s in schemes]):
                columns.append("Max")
            if any([schemes[s].labels is not None for s in schemes]):
                columns.append("Labels")
            if any([isinstance(schemes[s].labels, (str, dict)) for s in schemes]):
                columns.append("Mappings")

        return columns

    def _scheme_to_list(self, scheme_id):
        db = self.header
        scheme_info = self._scheme_table_columns

        scheme = db.schemes[scheme_id]

        data_dict = {
            "ID": scheme_id,
            "Dtype": scheme.dtype,
        }
        data = [scheme_id, scheme.dtype]
        #  minimum, maximum, labels, mappings = "", "", "", ""

        minimum, maximum = None, None
        labels = None

        if "Min" in scheme_info:
            minimum = scheme.minimum
            if minimum is None:
                minimum = ""
            data_dict["Min"] = minimum
        if "Max" in scheme_info:
            maximum = scheme.maximum
            if maximum is None:
                maximum = ""
            data_dict["Max"] = maximum
        if "Labels" in scheme_info:
            if scheme.labels is None:
                labels = []
            else:
                labels = sorted(scheme._labels_to_list())
                labels = [str(label) for label in labels]
                # Avoid `_` at end of label,
                # as this has special meaning in RST (link)
                labels = [
                    label[:-1] + r"\_" if label.endswith("_") else label
                    for label in labels
                ]
                labels = utils.limit_presented_samples(
                    labels,
                    15,
                    replacement_text="[...]",
                )
                labels = ", ".join(labels)
            data_dict["Labels"] = labels

        data.append(minimum)
        data.append(maximum)
        data.append(labels)
        if "Mappings" in scheme_info:
            if not isinstance(scheme.labels, (str, dict)):
                mappings = ""
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
                    mappings = f"{', '.join(mappings)}"
                else:
                    # e.g. {'f': 'female', 'm': 'male'}
                    mappings = "✓"

            data.append(mappings)
            data_dict["Mappings"] = mappings

        return data_dict

    @functools.cached_property
    def _segments(self) -> pa.Table:
        """Unique segments of all segmented tables of dataset.

        For tables stored as PARQUET files,
        only the index columns are read,
        and the tables are not downloaded
        for MinIO/S3 and file-system backends.
        Tables stored as CSV files
        are loaded completely.

        To reduce memory consumption
        for datasets with millions of segments,
        files are represented by integer codes
        instead of file names,
        and duplicated segments
        are removed by sorting integer arrays.

        Returns:
            table with columns ``"file"``, ``"start"``, ``"end"``,
            holding every segment once,
            in order of first occurrence.
            ``"file"`` holds an integer code per file

        """
        file_column = audformat.define.IndexField.FILE
        start_column = audformat.define.IndexField.START
        end_column = audformat.define.IndexField.END
        index_columns = [file_column, start_column, end_column]

        # Index columns of segmented tables,
        # with dictionary encoded file column
        indices = []
        for table in self.header.tables:
            if not self.header.tables[table].is_segmented:
                continue
            if f"db.{table}.parquet" in self.deps.tables:
                with self._open_parquet_table(table) as file:
                    parquet_file = pq.ParquetFile(file, read_dictionary=[file_column])
                    index = parquet_file.read(columns=index_columns)
            else:
                df = audb.load_table(
                    self.name,
                    table,
                    version=self.version,
                    verbose=False,
                )
                index = pa.Table.from_pandas(
                    df.index.to_frame(index=False),
                    preserve_index=False,
                )
                del df
                index = index.set_column(
                    index.schema.get_field_index(file_column),
                    file_column,
                    pc.dictionary_encode(index[file_column]),
                )
            indices.append(index.select(index_columns))

        # Map dictionary indices of all tables
        # to codes of a single list of files
        dictionaries = [
            chunk.dictionary.cast(pa.string())
            for index in indices
            for chunk in index[file_column].chunks
        ]
        if len(dictionaries) > 0:
            files = pc.unique(pa.concat_arrays(dictionaries))
        codes, starts, ends = [], [], []
        for index in indices:
            for chunk in index[file_column].chunks:
                mapping = pc.index_in(chunk.dictionary.cast(pa.string()), files)
                codes.append(pc.take(mapping, chunk.indices).to_numpy())
            starts.append(self._duration_to_int(index[start_column]))
            ends.append(self._duration_to_int(index[end_column]))
        del indices
        codes = np.concatenate(codes or [np.array([], dtype=np.int32)])
        starts = np.concatenate(starts or [np.array([], dtype=np.int64)])
        ends = np.concatenate(ends or [np.array([], dtype=np.int64)])

        # Remove duplicates,
        # and keep order of first occurrence.
        # As np.lexsort() is stable,
        # the first entry of identical segments
        # is the one with the lowest position
        if len(codes) > 0:
            order = np.lexsort((ends, starts, codes))
            keep = np.ones(len(order), dtype=bool)
            keep[1:] = (
                (np.diff(codes[order]) != 0)
                | (np.diff(starts[order]) != 0)
                | (np.diff(ends[order]) != 0)
            )
            keep = np.sort(order[keep])
            codes = codes[keep]
            starts = starts[keep]
            ends = ends[keep]

        null = np.iinfo(np.int64).min
        return pa.table(
            {
                file_column: pa.array(codes, type=pa.int32()),
                start_column: pa.array(starts).cast(pa.duration("ns")),
                end_column: pa.array(ends, mask=ends == null).cast(pa.duration("ns")),
            }
        )

    @staticmethod
    def _duration_to_int(column: pa.ChunkedArray) -> np.ndarray:
        r"""Convert start or end column to integer array in nanoseconds.

        Missing values are set to the minimum value of ``np.int64``.

        Args:
            column: column holding durations

        Returns:
            array with durations in nanoseconds

        """
        column = pc.cast(pc.cast(column, pa.duration("ns")), pa.int64())
        return pc.fill_null(column, np.iinfo(np.int64).min).to_numpy()

    @functools.cached_property
    def _tables_stats(self) -> dict[str, dict]:
        """Table information of tables in the dataset.

        Caches table information to improve performance
        of multiple table-related properties.
        This property computes and stores statistics for all tables,
        reducing repeated computations.
        It significantly improves performance
        when accessing multiple table properties frequently.

        Tables stored as PARQUET files
        are not loaded completely into memory,
        and are not downloaded
        for MinIO/S3 and file-system backends,
        as they might be too large.
        Instead,
        the number of rows is read from the file metadata,
        and only the first rows are read for the preview.
        Tables stored as CSV files
        are loaded completely.

        Returns:
            A dictionary with table names as keys and dictionaries containing:
            - "columns": number of columns
            - "rows": number of rows
            - "preview": dataframe preview (first 5 rows)

        """
        preview_rows = 5
        stats = {}
        # Header without loaded tables,
        # used to convert the preview of PARQUET tables
        # to a dataframe.
        # We cannot use `self.header` here,
        # as it holds already loaded misc tables,
        # which would be replaced by their preview.
        # For the same reason,
        # the previews of misc tables are loaded into their own header,
        # as other tables might use them as scheme labels,
        # and replacing them by their preview
        # would set all labels not contained in the preview to NaN.
        db = None
        for table in list(self.header):
            if f"db.{table}.parquet" in self.deps.tables:
                if table in self.header.misc_tables:
                    table_db = self._header_without_tables()
                else:
                    if db is None:
                        db = self._header_without_tables()
                    table_db = db
                rows, df = self._parquet_table_preview(table_db, table, preview_rows)
            else:
                df = audb.load_table(
                    self.name,
                    table,
                    version=self.version,
                    verbose=False,
                )
                rows = len(df)
                df = df.head(preview_rows)
            stats[table] = {
                "columns": len(df.columns),
                "rows": rows,
                "preview": df,
            }
        return stats

    @contextlib.contextmanager
    def _open_parquet_table(
        self,
        table: str,
    ) -> typing.Iterator[str | typing.BinaryIO]:
        r"""Open PARQUET file of table for random access reading.

        For backends supporting
        :meth:`audbackend.interface.Versioned.open_file`,
        e.g. MinIO/S3 and file-system backends,
        the file is accessed directly on the backend,
        without downloading it.
        For other backends,
        the PARQUET file is downloaded to the audb cache.

        Args:
            table: table ID

        Yields:
            binary file object or path to file

        """
        table_file = f"db.{table}.parquet"
        if isinstance(self.backend, audbackend.interface.Versioned):
            remote_file = self.backend.join("/", self.name, "meta", f"{table}.parquet")
            with self.backend.backend:
                try:
                    file = self.backend.open_file(
                        remote_file,
                        self.deps.version(table_file),
                    )
                except audbackend.BackendError as ex:
                    # Backend does not support random access to files
                    if not isinstance(ex.exception, NotImplementedError):
                        raise
                else:
                    with file:
                        yield file
                    return
        # Download table to audb cache
        stream = audb.stream(
            self.name,
            table,
            version=self.version,
            only_metadata=True,
            full_path=False,
            verbose=False,
        )
        yield os.path.join(stream.root, table_file)

    def _parquet_table_preview(
        self,
        db: audformat.Database,
        table: str,
        preview_rows: int,
    ) -> tuple[int, pd.DataFrame]:
        r"""Number of rows and preview of a table stored as PARQUET file.

        Only the footer
        and the first pages of the PARQUET file
        are read,
        see :meth:`_open_parquet_table`.

        Args:
            db: database header without loaded tables
            table: table ID
            preview_rows: number of rows of the preview

        Returns:
            number of rows of table,
            and dataframe holding the first ``preview_rows`` rows

        """
        with self._open_parquet_table(table) as file:
            rows, preview = utils.parquet_preview(file, preview_rows)
        # Store preview as PARQUET file
        # and load it with audformat,
        # to ensure the dataframe has the correct index and dtypes
        with tempfile.TemporaryDirectory() as tmp_root:
            preview_file = os.path.join(tmp_root, f"db.{table}")
            pq.write_table(preview, f"{preview_file}.parquet")
            db[table].load(preview_file)
        return rows, db[table].df

    @staticmethod
    def _map_iso_languages(languages: list[str]) -> list[str]:
        r"""Calculate ISO languages for a list of languages.

        Leaves languages intact if :func:`audformat.utils.map_language`
        raises :exception:`ValueError`.

        Args:
            languages: list of languages as given in the header languages

        Returns:
            list of languages

        """
        iso_languages = []
        for lang in languages:
            try:
                iso_language = audformat.utils.map_language(lang)
            except ValueError:
                iso_language = lang

            iso_languages.append(iso_language)

        return sorted(list(set(iso_languages)))


class Dataset(object):
    r"""Dataset representation.

    Dataset object that represents a dataset
    that can be loaded with :func:`audb.load()`.

    Args:
        name: name of dataset
        version: version of dataset
        cache_root: cache folder.
            If ``None``,
            the environmental variable ``AUDBCARDS_CACHE_ROOT``,
            or :attr:`audbcards.config.CACHE_ROOT`
            is used
        load_tables: if ``True``,
            it caches values extracted from tables.
            Set this to ``False``,
            if loading the tables takes too long,
            or does not fit into memory

    """

    def __new__(
        cls,
        name: str,
        version: str,
        *,
        cache_root: str | None = None,
        load_tables: bool = True,
    ):
        r"""Create Dataset Instance."""
        instance = _Dataset.create(
            name,
            version,
            cache_root=cache_root,
            load_tables=load_tables,
        )
        return instance

    # Add an __init__() function,
    # to allow documenting instance variables
    def __init__(
        self,
        name: str,
        version: str,
        *,
        cache_root: str | None = None,
        load_tables: bool = True,
    ):
        self.cache_root = audeer.mkdir(cache_root)
        r"""Cache root folder."""

    # Copy attributes and methods
    # to include in documentation
    for _prop in [  # use private variable `_prop` to avoid inclusion in API doc
        name
        for name, value in inspect.getmembers(_Dataset)
        if not name.startswith("_") and name not in ["create"]
    ]:
        vars()[_prop] = getattr(_Dataset, _prop)

    @staticmethod
    def _map_iso_languages(*args):
        return _Dataset._map_iso_languages(*args)

    @staticmethod
    def _dataset_cache_path(*args):
        cache_path = _Dataset._dataset_cache_path(*args)
        return cache_path

    @staticmethod
    def _load_pickled(path: str):
        ds = _Dataset._load_pickled(path)
        return ds

    @staticmethod
    def _parse_text(text: str) -> str:
        return _Dataset._parse_text(text)

    @staticmethod
    def _save_pickled(obj, path: str):
        """Save object instance to path as pickle."""
        return _Dataset._save_pickled(obj, path)


def create_datasets_page(
    datasets: Sequence[Dataset],
    rst_file: str = "./datasets.rst",
    *,
    datacards_path: str = "./datasets",
    header: str = "Datasets",
):
    r"""Create overview page of datasets.

    Writes the RST file ``rst_file`` to disk
    accompanied by a CSV with the same basename.
    The RST includes the CSV file
    as a table listing all available datasets
    containing the columns
    name,
    description,
    license,
    version,
    schemes.
    The name column does also contain a link
    to the corresponding data card.

    Args:
        datasets: list of datasets
        rst_file: name of RST file written to disk.
            Besides the RST file,
            a CSV file with the same basename
            is also stored
        datacards_path: relative path to folder that stores
            data cards for the given datasets
        header: header of the created RST file

    """
    tuples = [
        (
            f"`{dataset.name} <{datacards_path}/{dataset.name}.html>`__",
            dataset.short_description,
            f"`{dataset.license} <{dataset.license_link}>`__",
            dataset.version,
            dataset.schemes_summary,
        )
        for dataset in datasets
    ]
    df = pd.DataFrame.from_records(
        tuples,
        columns=["name", "description", "license", "version", "schemes"],
        index="name",
    )
    csv_file = audeer.replace_file_extension(rst_file, "csv")
    df.to_csv(csv_file)

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
    )
    template = environment.get_template("datasets.j2")

    data = [
        (
            dataset.name,
            dataset.version,
        )
        for dataset in datasets
    ]
    repositories = [f"{repo.name}" for repo in audb.config.REPOSITORIES]
    content = {
        "data": data,
        "name": audeer.basename_wo_ext(rst_file),
        "path": datacards_path,
        "header": header,
        "repositories": repositories,
    }
    content = template.render(content)

    with open(rst_file, mode="w", encoding="utf-8") as fp:
        fp.write(content)
