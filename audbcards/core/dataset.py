import datetime
import os
import random
import typing

import jinja2
import numpy as np
import pandas as pd
import toml

import audb
import audeer
import audfactory
import audformat

from audbcards.core.utils import format_schemes
from audbcards.core.utils import limit_presented_samples


class Dataset:
    r"""Dataset.

    Dataset object that represents a dataset
    that can be loaded with :func:`audb.load()`.

    Args:
        name: name of dataset
        version: version of dataset
        cache_root: cache folder

    """
    def __init__(
            self,
            name: str,
            version: str,
            cache_root: str = './cache',
    ):
        self._version = version
        self._repository = audb.repository(name, version)
        self.cache_root = audeer.mkdir(audeer.path(cache_root))

        self.header = audb.info.header(
            name,
            version=version,
            load_tables=True,  # ensure misc tables are loaded
            cache_root=self.cache_root,
        )
        self.deps = audb.dependencies(
            name,
            version=version,
            cache_root=self.cache_root,
            verbose=False,
        )

        # Clean up cache
        # by removing all other versions of the same dataset
        # to reduce its storage size in CI runners
        versions = audeer.list_dir_names(
            audeer.path(self.cache_root, name),
            basenames=True,
        )
        other_versions = [v for v in versions if v != version]
        for other_version in other_versions:
            audeer.rmdir(
                audeer.path(self.cache_root, name, other_version)
            )

    @property
    def archives(self) -> int:
        r"""Number of archives of media files in dataset."""
        return len(
            set([self.deps.archive(file) for file in self.deps.media])
        )

    @property
    def author(self) -> typing.List[str]:
        r"""Authors of the database."""
        return self.header.author

    @property
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

    @property
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

    @property
    def columns(self) -> typing.List[str]:
        """Columns of the dataset."""
        db = self.header
        columns = [list(db[table_id].columns) for table_id in self.tables]
        columns = [x for x in map(", ".join, columns)]
        return columns

    @property
    def dataset_schemes(self) -> list:
        """Dataset schemes with more information.

        Eache scheme is returned as a list
        containing its name, type, min, max, labels, mappings.

        """
        db = self.header
        dataset_schemes = []
        for scheme_id in db.schemes:
            dataset_scheme = self._scheme_to_list(scheme_id)
            dataset_schemes.append(dataset_scheme)

        cols = self.scheme_info['columns']
        data = pd.DataFrame.from_dict(dataset_schemes)[cols]
        filter = data.applymap(lambda d: d == [])
        data.mask(filter, other='', inplace=True)
        scheme_data = np.array(data).tolist()
        return scheme_data

    @property
    def description(self) -> str:
        r"""Source of the database."""
        return self.header.description

    @property
    def duration(self) -> pd.Timedelta:
        r"""Total duration of media files in dataset."""
        durations = [self.deps.duration(file) for file in self.deps.media]
        return pd.to_timedelta(
            sum([d for d in durations if d is not None]),
            unit='s',
        )

    @property
    def example(self) -> str:
        r"""Relative path to example media file in dataset."""
        # Pick a meaningful duration for the example audio file
        min_dur = 0.5
        max_dur = 300  # 5 min
        durations = [self.deps.duration(file) for file in self.deps.media]
        selected_duration = np.median(
            [d for d in durations if d >= min_dur and d <= max_dur]
        )
        # Get index for duration closest to selected duration
        # see https://stackoverflow.com/a/9706105
        # durations.index(selected_duration)
        # is an alternative but fails due to rounding errors
        index = min(
            range(len(durations)),
            key=lambda n: abs(durations[n] - selected_duration),
        )
        # Download of example data might fail
        try:
            media = self.deps.media[index]
            audb.load_media(
                self.name,
                media,
                version=self._version,
                cache_root=self.cache_root,
                verbose=False,
            )
        except:  # noqa: E722
            media = ''
        return media

    @property
    def files(self) -> int:
        r"""Number of media files in dataset."""
        return len(self.deps.media)

    @property
    def formats(self) -> typing.List[str]:
        r"""File formats of media files in dataset."""
        return sorted(
            list(
                set(
                    [
                        self.deps.format(file)
                        for file in self.deps.media
                    ]
                )
            )
        )

    @property
    def languages(self) -> typing.List[str]:
        r"""Languages of the database."""
        return self.header.languages

    @property
    def license(self) -> str:
        r"""License of dataset.

        If no license is given,
        ``'Unknown'`` is returned.

        """
        return self.header.license or 'Unknown'

    @property
    def license_link(self) -> typing.Optional[str]:
        r"""Link to license of dataset.

        If no link is available
        ``None`` is returned.

        """
        if (
                self.header.license_url is None
                or len(self.header.license_url) == 0
        ):
            return None
        else:
            return self.header.license_url

    @property
    def name(self) -> str:
        r"""Name of dataset."""
        return self.header.name

    @property
    def publication_date(self) -> str:
        r"""Date dataset was uploaded to repository."""
        # NOTE: the following code can be replaced
        # by audbackend.Backend.date()
        # when audbackend 1.0.0 is released
        url = (
            f'{self._repository.host}/{self._repository.name}/{self.name}/'
            f'db/{self._version}/db-{self._version}.zip'
        )

        if self._repository.backend == 'file-system':
            ts = os.stat(url).st_ctime
            date = datetime.datetime.utcfromtimestamp(ts)
            date = date.strftime("%Y-%m-%d")
        else:
            path = audfactory.path(url)
            stat = path.stat()
            date = f'{stat.ctime:%Y-%m-%d}'

        return date

    @property
    def publication_owner(self) -> str:
        r"""User who uploaded dataset to repository."""
        # NOTE: the following code can be replaced
        # by audbackend.Backend.owner()
        # when audbackend 1.0.0 is released
        url = (
            f'{self._repository.host}/{self._repository.name}/{self.name}/'
            f'db/{self._version}/db-{self._version}.zip'
        )

        if self._repository.backend == 'file-system':
            # NOTE: the following will
            config = toml.load(audeer.path('pyproject.toml'))
            authors = ', '.join(
                author['name']
                for author in config['project']['authors']
            )
            owners = authors.split(', ')
            owner = random.choice(owners)
        else:
            path = audfactory.path(url)
            stat = path.stat()
            owner = f'{stat.created_by}'

        return owner

    def properties(self):
        """Get list of properties of the object."""
        class_items = self.__class__.__dict__.items()
        props = dict((k, getattr(self, k))
                     for k, v in class_items
                     if isinstance(v, property))

        return props

    @property
    def repository(self) -> str:
        r"""Repository containing the dataset."""
        return f'{self._repository.name}'

    @property
    def repository_link(self) -> str:
        r"""Link to repository in Artifactory web UI."""
        # NOTE: this needs to be changed
        # as we want to support different backends
        return (
            f'{self._repository.host}/'
            f'webapp/#/artifacts/browse/tree/General/'
            f'{self._repository.name}/'
            f'{self.name}'
        )

    @property
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

    @property
    def scheme_info(self) -> dict:
        """Information on schemes."""
        db = self.header
        scheme_info = {}

        if len(db.schemes) > 0:
            has_minimums = any(
                [db.schemes[s].minimum is not None for s in db.schemes]
            )
            has_maximums = any(
                [db.schemes[s].maximum is not None for s in db.schemes]
            )
            has_labels = any(
                [db.schemes[s].labels is not None for s in db.schemes]
            )
            has_mappings = any(
                [
                    isinstance(db.schemes[s].labels, (str, dict))
                    for s in db.schemes
                ]
            )

        columns = ['ID', 'Dtype']
        header_line = '   :header: ID,Dtype'
        if has_minimums:
            header_line += ',Min'
            columns.append('Min')
        if has_maximums:
            header_line += ',Max'
            columns.append('Max')
        if has_labels:
            header_line += ',Labels'
            columns.append('Labels')
        if has_mappings:
            header_line += ',Mappings'
            columns.append('Mappings')

        scheme_info['header_line'] = header_line
        scheme_info['has_minimums'] = has_minimums
        scheme_info['has_maximums'] = has_maximums
        scheme_info['has_labels'] = has_labels
        scheme_info['has_mappings'] = has_mappings

        scheme_info['columns'] = columns
        return scheme_info

    @property
    def schemes(self) -> typing.List[str]:
        r"""Schemes of dataset."""
        return list(self.header.schemes)

    @property
    def short_description(self) -> str:
        r"""Description of dataset shortened to 150 chars."""
        length = 150
        description = self.header.description or ''
        # Fix RST used signs
        description = description.replace('`', "'")
        if len(description) > length:
            description = f'{description[:length - 3]}...'
        return description

    @property
    def source(self) -> str:
        r"""Source of the database."""
        return self.header.source

    @property
    def tables(self) -> typing.List[str]:
        """Tables of the dataset."""
        db = self.header
        tables = list(db)
        return tables

    @property
    def types(self) -> typing.List[str]:
        """Table types of the dataset."""
        types = []
        db = self.header
        for table_id in self.tables:
            table = db[table_id]
            if isinstance(table, audformat.MiscTable):
                types.append('misc')
            else:
                types.append(table.type)

        return types

    @property
    def usage(self) -> str:
        r"""Usage of the database."""
        return self.header.usage

    @property
    def version(self) -> str:
        r"""Version of dataset."""
        return self._version

    def _scheme_to_list(self, scheme_id):

        db = self.header
        scheme_info = self.scheme_info

        scheme = db.schemes[scheme_id]

        data_dict = {
            'ID': scheme_id,
            'Dtype': scheme.dtype,
        }
        data = [scheme_id, scheme.dtype]
        #  minimum, maximum, labels, mappings = "", "", "", ""

        minimum, maximum = None, None
        labels = None

        # can use 'Minimum' in scheme_info['columns'] later on
        if scheme_info["has_minimums"]:
            minimum = scheme.minimum or ''
            data_dict['Min'] = minimum
        if scheme_info["has_maximums"]:
            maximum = scheme.maximum or ''
            data_dict['Max'] = maximum
        if scheme_info["has_labels"]:
            if scheme.labels is None:
                labels = []
            else:
                labels = sorted(scheme._labels_to_list())
                labels = [str(label) for label in labels]
                # Avoid `_` at end of label,
                # as this has special meaning in RST (link)
                labels = [
                    label[:-1] + r'\_'
                    if label.endswith('_')
                    else label
                    for label in labels
                ]
                labels = limit_presented_samples(
                    labels,
                    15,
                    replacement_text='[...]',
                )
                labels = ", ".join(labels)
            scheme_info['Labels'] = labels

        data.append(minimum)
        data.append(maximum)
        data.append(labels)
        data_dict['Labels'] = labels
        if scheme_info["has_mappings"]:
            if not isinstance(scheme.labels, (str, dict)):
                mappings = ''
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
                    mappings = '✓'

            data.append(mappings)
            data_dict['Mappings'] = mappings

        return data_dict


def create_datasets_page(
        datasets: typing.Sequence[Dataset],
        ofbase: str = 'datasets',
):
    r"""Create overview page of datasets.

    Args:
        datasets: list of datasets
        ofbase: basename of the file written to disk

    ofbase: written to disk in both csv and rst formats.
    Final outfilenames consist of ofbase plus extension.

    """
    tuples = [
        (
            f'`{dataset.name} <./datasets/{dataset.name}.html>`__',
            dataset.short_description,
            dataset.license_link,
            dataset.version,
            format_schemes(dataset.header.schemes),
        )
        for dataset in datasets
    ]
    df = pd.DataFrame.from_records(
        tuples,
        columns=['name', 'description', 'license', 'version', 'schemes'],
        index='name',
    )
    csv_file = f'{ofbase}.csv'
    df.to_csv(csv_file)

    rst_file = f'{ofbase}.rst'

    t_dir = os.path.join(os.path.dirname(__file__), 'templates')
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(t_dir),
                                     trim_blocks=True)

    data = [
        (
            dataset.name,
            dataset.version,
        )
        for dataset in datasets
    ]

    data = {
        'data': data,
    }

    template = environment.get_template("datasets.j2")
    content = template.render(data)

    with open(rst_file, mode="w", encoding="utf-8") as fp:
        fp.write(content)
        print(f"... wrote {rst_file}")
