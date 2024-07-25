import functools
import inspect
import os
import pickle
import re
import typing

import jinja2
import numpy as np
import pandas as pd

import audb
import audbackend
import audeer
import audformat

from audbcards.core import utils
from audbcards.core.config import config


class _Dataset:
    _table_related_cached_properties = [
        "segment_durations",
        "segments",
        "tables_preview",
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
        cache_root: str = None,
        load_tables: bool = True,
    ):
        r"""Instantiate Dataset Object."""
        if cache_root is None:
            cache_root = os.environ.get("AUDBCARDS_CACHE_ROOT") or config.CACHE_ROOT
        dataset_cache_filename = cls._dataset_cache_path(name, version, cache_root)

        if os.path.exists(dataset_cache_filename):
            obj = cls._load_pickled(dataset_cache_filename)
            # Load cached properties,
            # that require to load filewise or segmented tables,
            # if they haven't been cached before.
            if load_tables:
                cache_again = False
                for cached_property in cls._table_related_cached_properties:
                    # Check if property has been cached,
                    # see https://stackoverflow.com/a/59740750
                    if cached_property not in obj.__dict__:
                        cache_again = True
                        # Request property to fill their cached value
                        getattr(obj, cached_property)
                if cache_again:
                    # Update cache to store the table related cached properties
                    cls._save_pickled(obj, dataset_cache_filename)

            return obj

        obj = cls(name, version, cache_root, load_tables)
        # Visit cached properties to fill their cache values
        if load_tables:
            exclude = []
        else:
            exclude = cls._table_related_cached_properties
        obj._cached_properties(exclude=exclude)

        cls._save_pickled(obj, dataset_cache_filename)
        return obj

    def __init__(
        self,
        name: str,
        version: str,
        cache_root: str = None,
        load_tables: bool = True,
    ):
        self.cache_root = audeer.mkdir(cache_root)
        r"""Cache root folder."""

        # Define `__getstate__()` method,
        # which selects the cached attributes
        # to include in the pickled cache file
        if load_tables:
            exclude = []
        else:
            exclude = self._table_related_cached_properties

        def getstate():
            return self._cached_properties(exclude=exclude)

        self.__getstate__ = getstate

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
        """Save object instance to path as pickle."""
        audeer.mkdir(os.path.dirname(path))

        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)

    @property
    def backend(self) -> typing.Type[audbackend.interface.Base]:
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
    def author(self) -> typing.List[str]:
        r"""Authors of the database."""
        return self.header.author

    @functools.cached_property
    def bit_depths(self) -> typing.List[int]:
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
    def channels(self) -> typing.List[int]:
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
    def example_media(self) -> typing.Optional[str]:
        r"""Example media file.

        The media file is selected
        by its median duration
        from all files in the dataset
        with a duration
        between 0.5 s and 300 s.
        In addition,
        the media file needs to be stored in an archive
        with less than 100 media files.
        If no media file meets this criterium,
        ``None`` is returned instead.

        """
        # Pick a meaningful duration for the example audio file
        min_dur = 0.5
        max_dur = 300  # 5 min
        durations = self.file_durations
        selected_durations = [d for d in durations if d >= min_dur and d <= max_dur]

        if len(selected_durations) == 0:
            return None

        selected_duration = np.median(selected_durations)
        # Get index for duration closest to selected duration
        # see https://stackoverflow.com/a/9706105
        # durations.index(selected_duration)
        # is an alternative but fails due to rounding errors
        index = min(
            range(len(durations)),
            key=lambda n: abs(durations[n] - selected_duration),
        )
        # Ensure we don't have too many other files in the archive
        # containing the selected file
        archives = self.deps().archive
        selected_archive = archives.iloc[index]
        number_of_files = len(
            [archive for archive in archives if archive == selected_archive]
        )
        selected_media = None
        if number_of_files < 100:
            selected_media = self.deps.media[index]
        return selected_media

    @functools.cached_property
    def files(self) -> int:
        r"""Number of media files in dataset."""
        return len(self.deps.media)

    @functools.cached_property
    def file_durations(self) -> typing.List:
        r"""File durations in dataset in seconds."""
        return [self.deps.duration(file) for file in self.deps.media]

    @functools.cached_property
    def formats(self) -> typing.List[str]:
        r"""File formats of media files in dataset."""
        return sorted(list(set([self.deps.format(file) for file in self.deps.media])))

    @functools.cached_property
    def languages(self) -> typing.List[str]:
        r"""Languages of the database."""
        return self.header.languages

    @functools.cached_property
    def iso_languages(self) -> typing.List[str]:
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
    def license_link(self) -> typing.Optional[str]:
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
    def sampling_rates(self) -> typing.List[int]:
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
    def schemes(self) -> typing.List[str]:
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
    def schemes_table(self) -> typing.List[typing.List[str]]:
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
        return str(len(self._segments))

    @functools.cached_property
    def segment_durations(self) -> typing.List[float]:
        r"""Segment durations in dataset."""
        if len(self._segments) == 0:
            durations = []
        else:
            starts = self._segments.get_level_values("start")
            ends = self._segments.get_level_values("end")
            durations = [
                (end - start).total_seconds() for start, end in zip(starts, ends)
            ]
        return durations

    @functools.cached_property
    def short_description(self) -> str:
        r"""Description of dataset shortened to 150 chars."""
        length = 150
        description = self.header.description or ""
        # Fix RST used signs
        description = description.replace("`", "'")
        if len(description) > length:
            description = f"{description[:length - 3]}..."
        return description

    @functools.cached_property
    def source(self) -> str:
        r"""Source of the database."""
        return self.header.source

    @functools.cached_property
    def tables(self) -> typing.List[str]:
        """Tables of the dataset."""
        db = self.header
        tables = list(db)
        return tables

    @functools.cached_property
    def tables_preview(self) -> typing.Dict[str, typing.List[typing.List[str]]]:
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
        for table in list(self.header):
            df = audb.load_table(
                self.name,
                table,
                version=self.version,
                verbose=False,
            )
            df = df.reset_index()
            header = [df.columns.tolist()]
            body = df.head(5).astype("string").values.tolist()
            # Remove unwanted chars and limit length of each entry
            body = [[self._parse_text(column) for column in row] for row in body]
            preview[table] = header + body
        return preview

    @functools.cached_property
    def tables_table(self) -> typing.List[str]:
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
        exclude: typing.Sequence = [],
    ) -> typing.Dict[str, typing.Any]:
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

    def _load_backend(self) -> typing.Type[audbackend.interface.Base]:
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

    def _load_repository_object(self) -> audb.Repository:
        r"""Load repository object containing dataset."""
        return audb.repository(self.name, self.version)

    @functools.cached_property
    def _scheme_table_columns(self) -> typing.List[str]:
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
                    mappings = f'{", ".join(mappings)}'
                else:
                    # e.g. {'f': 'female', 'm': 'male'}
                    mappings = "✓"

            data.append(mappings)
            data_dict["Mappings"] = mappings

        return data_dict

    @functools.cached_property
    def _segments(self) -> pd.MultiIndex:
        """Segments of dataset as combined index."""
        index = audformat.segmented_index()
        for table in self.header.tables:
            if self.header.tables[table].is_segmented:
                df = audb.load_table(
                    self.name,
                    table,
                    version=self.version,
                    verbose=False,
                )
                index = audformat.utils.union([index, df.index])
        return index

    @staticmethod
    def _map_iso_languages(languages: typing.List[str]) -> typing.List[str]:
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
        cache_root: str = None,
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
        cache_root: str = None,
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
    datasets: typing.Sequence[Dataset],
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
    repositories = [
        f"`{repo.name} <{repo.host}>`__" for repo in audb.config.REPOSITORIES
    ]
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
