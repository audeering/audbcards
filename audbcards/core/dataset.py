import functools
import inspect
import os
import pickle
import typing

import dohq_artifactory
import jinja2
import pandas as pd

import audb
import audbackend
import audeer
import audformat

from audbcards.core.utils import format_schemes
from audbcards.core.utils import limit_presented_samples


def _getstate(self):
    return self.name


def _setstate(self, state):
    self.name = state


# Ensure we can pickle the repository
dohq_artifactory.GenericRepository.__getstate__ = _getstate
dohq_artifactory.GenericRepository.__setstate__ = _setstate


class _Dataset:
    @classmethod
    def create(
        cls,
        name: str,
        version: str,
        *,
        cache_root: str = "~/.cache/audbcards",
    ):
        r"""Instantiate Dataset Object."""
        dataset_cache_filename = cls._dataset_cache_path(name, version, cache_root)

        if os.path.exists(dataset_cache_filename):
            obj = cls._load_pickled(dataset_cache_filename)

            return obj

        obj = cls(name, version, cache_root)
        _ = obj.properties()

        cls._save_pickled(obj, dataset_cache_filename)
        return obj

    def __init__(
        self,
        name: str,
        version: str,
        cache_root: str = "~./cache/audbcards",
    ):
        self.cache_root = audeer.mkdir(audeer.path(cache_root))
        self.header = audb.info.header(
            name,
            version=version,
            load_tables=True,  # ensure misc tables are loaded
        )
        self.deps = audb.dependencies(
            name,
            version=version,
            verbose=False,
        )

        self._version = version
        self._repository = audb.repository(name, version)
        self._backend = audbackend.access(
            name=self._repository.backend,
            host=self._repository.host,
            repository=self._repository.name,
        )
        if isinstance(self._backend, audbackend.Artifactory):
            self._backend._use_legacy_file_structure()  # pragma: nocover

        # Clean up cache
        # by removing all other versions of the same dataset
        # to reduce its storage size in CI runners
        versions = audeer.list_dir_names(
            audeer.path(self.cache_root, name),
            basenames=True,
        )
        other_versions = [v for v in versions if v != version]
        for other_version in other_versions:
            audeer.rmdir(audeer.path(self.cache_root, name, other_version))

    @staticmethod
    def _dataset_cache_path(name: str, version: str, cache_root: str) -> str:
        r"""Generate the name of the cache file."""
        cache_dir = audeer.mkdir(audeer.path(cache_root, name, version))

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
    def _save_pickled(obj, path: str):
        """Save object instance to path as pickle."""
        audeer.mkdir(os.path.dirname(path))

        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)

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
        return self.header.name

    @functools.cached_property
    def publication_date(self) -> str:
        r"""Date dataset was uploaded to repository."""
        path = self._backend.join("/", self.name, "db.yaml")
        return self._backend.date(path, self._version)

    @functools.cached_property
    def publication_owner(self) -> str:
        r"""User who uploaded dataset to repository."""
        path = self._backend.join("/", self.name, "db.yaml")
        return self._backend.owner(path, self._version)

    def properties(self):
        """Get list of properties of the object."""
        class_items = self.__class__.__dict__.items()
        props = dict(
            (k, getattr(self, k))
            for k, v in class_items
            if isinstance(v, functools.cached_property)
        )
        return props

    @functools.cached_property
    def repository(self) -> str:
        r"""Repository containing the dataset."""
        return f"{self._repository.name}"

    @functools.cached_property
    def repository_link(self) -> str:
        r"""Link to repository in Artifactory web UI."""
        # NOTE: this needs to be changed
        # as we want to support different backends
        return (
            f"{self._repository.host}/"
            f"webapp/#/artifacts/browse/tree/General/"
            f"{self._repository.name}/"
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
                labels = limit_presented_samples(
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
        cache_root: cache folder

    """

    def __new__(
        cls,
        name: str,
        version: str,
        *,
        cache_root: str = "~/.cache/audbcards",
    ):
        r"""Create Dataset Instance."""
        instance = _Dataset.create(name, version, cache_root=cache_root)
        return instance

    # Copy attributes and methods
    # to include in documentation
    for prop in [
        name
        for name, value in inspect.getmembers(_Dataset)
        if isinstance(value, functools.cached_property) and not name.startswith("_")
    ]:
        vars()[prop] = getattr(_Dataset, prop)

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
            format_schemes(dataset.header.schemes),
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
