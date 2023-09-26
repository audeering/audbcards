import configparser
import datetime
import functools
import os
import random
import shutil
import typing

import jinja2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import audb
import audeer
import audfactory
import audformat
import audiofile
import audplot

# Configuration -----------------------------------------------------------

BUILD = audeer.path('..', 'build', 'html')


# Functions to create data cards -------------------------------------------
class Dataset:

    def __init__(
            self,
            name: str,
            version: str,
            cache_root: str = './cache',
    ):
        self.name = name
        self.version = version
        self.repository = audb.repository(name, version)
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
    def archives(self) -> str:
        r"""Number of archives of media files in dataset."""
        return str(
            len(
                set([self.deps.archive(file) for file in self.deps.media])
            )
        )

    @property
    def bit_depths(self) -> str:
        r"""Bit depths of media files in dataset."""
        return ', '.join(
            set(
                [
                    str(self.deps.bit_depth(file))
                    for file in self.deps.media
                    if self.deps.bit_depth(file)
                ]
            )
        )

    @property
    def channels(self) -> str:
        r"""Channels of media files in dataset."""
        return ', '.join(
            set(
                [
                    str(self.deps.channels(file))
                    for file in self.deps.media
                    if self.deps.channels(file)
                ]
            )
        )

    @property
    def duration(self) -> str:
        r"""Total duration of media files in dataset."""
        durations = [self.deps.duration(file) for file in self.deps.media]
        return str(
            pd.to_timedelta(
                sum([d for d in durations if d is not None]),
                unit='s',
            )
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
                version=self.version,
                cache_root=self.cache_root,
                verbose=False,
            )
        except:  # noqa: E722
            media = ''
        return media

    @property
    def files(self) -> str:
        r"""Number of media files in dataset."""
        return str(len(self.deps.media))

    @property
    def formats(self) -> str:
        r"""File formats of media files in dataset."""
        return ', '.join(
            set(
                [
                    self.deps.format(file)
                    for file in self.deps.media
                ]
            )
        )

    @property
    def license_link(self) -> str:
        r"""License of dataset as a link if possible."""
        license = self.header.license or 'Unknown'
        if (
                self.header.license_url is None
                or len(self.header.license_url) == 0
        ):
            return license
        else:
            return f'`{license} <{self.header.license_url}>`__'

    @property
    def name_link(self) -> str:
        r"""Name of dataset as internal link to data card."""
        return f'`{self.name} <./datasets/{self.name}.html>`__'

    def player(
            self,
            file,
            *,
            waveform=True,
    ) -> str:
        r"""Create an audio player showing the waveform."""
        player_str = ''
        # Move file to build folder
        src_dir = (
            f'{self.cache_root}/'
            f'{audb.flavor_path(self.name, self.version)}'
        )
        dst_dir = f'{BUILD}/datasets/{self.name}'
        audeer.mkdir(os.path.join(dst_dir, os.path.dirname(file)))
        shutil.copy(
            os.path.join(src_dir, file),
            os.path.join(dst_dir, file),
        )
        if waveform:
            # Add plot of waveform
            signal, sampling_rate = audiofile.read(
                os.path.join(src_dir, file),
                always_2d=True,
            )
            plt.figure(figsize=[3, .5])
            ax = plt.subplot(111)
            audplot.waveform(signal[0, :], ax=ax)
            set_plot_margins()
            plt.savefig(f'{self.name}.png')
            plt.close()

            player_str += (
                f'.. image:: ../{self.name}.png\n'
                '\n'
            )
        player_str += (
            '.. raw:: html\n'
            '\n'
            f'    <p><audio controls src="{self.name}/{file}"></audio></p>'
        )
        return player_str

    @property
    def publication(self) -> str:
        r"""Date and author uploading dataset to repository."""
        url = (
            f'{self.repository.host}/{self.repository.name}/{self.name}/'
            f'db/{self.version}/db-{self.version}.zip'
        )

        if self.repository.backend == 'file-system':
            ts = os.stat(url).st_ctime
            date_created = datetime.datetime.utcfromtimestamp(ts)
            date_created = date_created.strftime("%Y-%m-%d")
            cf = configparser.ConfigParser()
            cf.read('setup.cfg')
            authors = cf['metadata']['Author']
            creators = authors.split(', ')
            creator = random.choice(creators)
            publication = f'{date_created} by {creator}'
        else:
            path = audfactory.path(url)
            stat = path.stat()
            publication = f'{stat.ctime:%Y-%m-%d} by {stat.created_by}'

        return publication

    @property
    def repository_link(self) -> str:
        r"""Repository name with link to dataset in Artifactory web UI."""
        url = (
            f'{self.repository.host}/'
            f'webapp/#/artifacts/browse/tree/General/'
            f'{self.repository.name}/'
            f'{self.name}'
        )
        return f'`{self.repository.name} <{url}>`__'

    @property
    def sampling_rates(self) -> str:
        r"""Sampling rates of media files in dataset."""
        return ', '.join(
            set(
                [
                    str(self.deps.sampling_rate(file))
                    for file in self.deps.media
                    if self.deps.sampling_rate(file)
                ]
            )
        )

    @property
    def schemes(self) -> str:
        r"""List schemes of dataset.

        For the schemes ``'speaker'`` and ``'emotion'``
        it incorporates mappings in brackets,
        e.g. ``'speaker: [age, gender, language]'``.

        """
        return format_schemes(self.header.schemes)

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
    def version_link(self) -> str:
        r"""Version of dataset as link to changelog on Github."""
        github = 'https://github.com/audeering'
        branch = 'main'
        url = f'{github}/{self.name}/blob/{branch}/CHANGELOG.md'
        return f'`{self.version} <{url}>`__'

    """Additions to class """
    def properties(self):
        """Get list of properties of the object."""

        class_items = self.__class__.__dict__.items()
        props = dict((k, getattr(self, k))
                     for k, v in class_items
                     if isinstance(v, property))

        props['name'] = self.name
        props['player'] = self.player(self.example)

        return props

    @property
    def source(self) -> str:
        r"""Source of the database."""
        return self.header.source

    @property
    def description(self) -> str:
        r"""Source of the database."""
        return self.header.description

    @property
    def usage(self) -> str:
        r"""Usage of the database."""
        return self.header.usage

    @property
    def languages(self) -> typing.List[str]:
        r"""Languages of the database."""
        return self.header.languages

    @property
    def author(self) -> typing.List[str]:
        r"""Authors of the database."""
        return self.header.author

    @property
    def tables(self) -> typing.List[str]:
        """List od Tables in db."""
        db = self.header
        tables = list(db)
        return tables

    @property
    def columns(self) -> typing.List[str]:
        db = self.header
        columns = [list(db[table_id].columns) for table_id in self.tables]
        columns = [x for x in map(", ".join, columns)]
        return columns

    @property
    def types(self) -> typing.List[str]:
        types = []
        db = self.header
        for table_id in self.tables:
            table = db[table_id]
            if isinstance(table, audformat.MiscTable):
                types.append('misc')
            else:
                types.append(table.type)

        return types

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

    @property
    def scheme_info(self) -> dict:

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
    def dataset_schemes(self) -> list:

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


class Datacard(object):

    def __init__(self, dataset: Dataset):

        self._dataset = dataset

    @property
    def props(self) -> dict:
        """slot to hold superclass properties for rendering."""
        props = self._dataset.properties()
        return props

    @functools.cached_property
    def content(self):
        """Property Accessor for rendered jinja2 content."""

        return self._render_template()

    def _render_template(self):

        t_dir = os.path.join(os.path.dirname(__file__), 'templates')
        environment = jinja2.Environment(loader=jinja2.FileSystemLoader(t_dir),
                                         trim_blocks=True)
        # Provide Jinja filter access to Python build-ins/functions
        environment.filters.update(
            zip=zip,
            tw=self._trim_trailing_whitespace,
        )
        template = environment.get_template("datacard.j2")
        content = template.render(self.props)
        return content

    @staticmethod
    def _trim_trailing_whitespace(x: list):
        """J2 filter to get rid of trailing empty table entries within a row.

        Trims last entry if present.

        Args:
            x: untrimmed single scheme table row
        Returns:
            trimmed single scheme table row
        """

        if x[-1] == '':
            x.pop()

        return x

    def save(self, ofpath: str = None):
        """Save content of rendered template to rst.

        Args:
            ofpath: filepath to save rendered template to
        Returns:
            None

        if ofpath is specified, the directory must exist.
        """

        if ofpath is None:
            ofpath = f'datasets/{self._dataset.name}_from_template.rst'

        with open(ofpath, mode="w", encoding="utf-8") as fp:
            fp.write(self.content)
            print(f"... wrote {ofpath}")


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
            dataset.name_link,
            dataset.short_description,
            dataset.license_link,
            dataset.version,
            dataset.schemes,
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


def format_schemes(
        schemes,
        excludes=['duration'],
        max_schemes=15,
):
    """Convert schemes object into string.

    It lists the main annotation schemes
    of the datasets,
    and collects additional information
    on schemes calls `emotion` and `speaker`.

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
        if scheme == 'emotion':
            try:
                labels = schemes[scheme]._labels_to_list()
                labels = audeer.flatten_list(labels)
                emotion = [{scheme: labels}]
                max_schemes -= 1
            except KeyError:
                emotion = [scheme]
        elif scheme == 'speaker':
            try:
                labels = schemes[scheme]._labels_to_dict()
                labels = list(labels.values())
                # Store the dictionary keys for speaker
                # as those are gender, language, ...
                # Keys are the same for all entries,
                # using the first one is enough
                labels = list(labels[0].keys())
                speaker = [{scheme: labels}]
                max_schemes -= 1
            except KeyError:
                emotion = [scheme]
            except AttributeError:
                emotion = [scheme]
        else:
            filtered_schemes.append(scheme)
    # Force emotion and speaker to the beginning of the list
    filtered_schemes = emotion + speaker + filtered_schemes
    # Limit to maximum number of schemes and add '...' for longer once
    max_schemes = max(max_schemes, 2)
    filtered_schemes = limit_presented_samples(filtered_schemes, max_schemes)
    # Format the information for display
    info_str = ''
    for scheme in filtered_schemes:
        if isinstance(scheme, dict):
            key = list(scheme.keys())[0]
            info_str += f'{key}: ['
            for label in scheme[key]:
                info_str += f'{label}, '
            info_str = info_str[:-2] + '], '
        else:
            info_str += f'{scheme}, '
    info_str = info_str[:-2]

    return info_str


def limit_presented_samples(
        samples: typing.Sequence,
        limit: int,
        replacement_text: str = '...',
) -> typing.List:
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
        samples = (
            samples[:limit // 2]
            + [replacement_text]
            + samples[-limit // 2:]
        )
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
