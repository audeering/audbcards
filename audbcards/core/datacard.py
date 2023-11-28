import functools
import os
import shutil
import typing

import jinja2
import matplotlib.pyplot as plt
import numpy as np

import audb
import audeer
import audiofile
import audplot

from audbcards.core.dataset import Dataset
from audbcards.core.utils import set_plot_margins


class Datacard(object):
    r"""Datacard.

    Datacard object to write a RST file
    for a given dataset.

    Args:
        dataset: dataset object
        path: path to folder
            that store datacard files

    """
    def __init__(
            self,
            dataset: Dataset,
            *,
            path: str = 'datasets',
    ):

        self.dataset = dataset
        """Dataset object."""

        self.path = path
        """Folder to store datacard."""

        self._sphinx_build_dir = None
        """Sphinx build dir."""

        self._sphinx_src_dir = None
        """Sphinx source dir."""

    @functools.cached_property
    def content(self):
        """Property Accessor for rendered jinja2 content."""
        return self._render_template()

    @property
    def example(self) -> typing.Optional[str]:
        r"""Select example media file.

        This select a media file
        based on the median duration
        of all files
        between 0.5 s and 300 s
        and downloads it to the cache.

        """
        # Pick a meaningful duration for the example audio file
        min_dur = 0.5
        max_dur = 300  # 5 min
        durations = self.dataset.file_durations
        selected_duration = np.median(
            [d for d in durations if d >= min_dur and d <= max_dur]
        )
        if np.isnan(selected_duration):
            return None
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
            media = self.dataset.deps.media[index]
            audb.load_media(
                self.dataset.name,
                media,
                version=self.dataset.version,
                cache_root=self.dataset.cache_root,
                verbose=False,
            )
        except:  # noqa: E722
            media = None
        return media

    def player(
            self,
            file: str,
    ) -> str:
        r"""Create an audio player showing the waveform.

        Args:
            file: input audio file to be used in the player.
                :attr:`audbcards.Datacard.example`
                is a good fit

        """
        media_src_dir = (
            f'{self.dataset.cache_root}/'
            f'{audb.flavor_path(self.dataset.name, self.dataset.version)}'
        )
        # Move file to build folder
        if self._sphinx_build_dir is not None:
            media_dst_dir = audeer.path(
                self._sphinx_build_dir,
                self.path,
                self.dataset.name,
            )
            audeer.mkdir(os.path.join(media_dst_dir, os.path.dirname(file)))
            shutil.copy(
                os.path.join(media_src_dir, file),
                os.path.join(media_dst_dir, file),
            )

        # Add plot of waveform
        if self._sphinx_src_dir is not None:
            signal, sampling_rate = audiofile.read(
                os.path.join(media_src_dir, file),
                always_2d=True,
            )
            image_file = audeer.path(
                self._sphinx_src_dir,
                self.path,
                self.dataset.name,
                f'{self.dataset.name}.png',
            )
            audeer.mkdir(os.path.dirname(image_file))
            plt.figure(figsize=[3, .5])
            ax = plt.subplot(111)
            audplot.waveform(signal[0, :], ax=ax)
            set_plot_margins()
            plt.savefig(image_file)
            plt.close()

        player_src = f'./{self.dataset.name}/{file}'
        player_str = (
            f'.. image:: ./{self.dataset.name}/{self.dataset.name}.png\n'
            '\n'
            '.. raw:: html\n'
            '\n'
            f'    <p><audio controls src="{player_src}"></audio></p>'
        )
        return player_str

    def save(self):
        """Save content of rendered template to rst."""
        if self._sphinx_src_dir is not None:
            rst_file = audeer.path(
                self._sphinx_src_dir,
                self.path,
                f'{self.dataset.name}.rst',
            )
            with open(rst_file, mode="w", encoding="utf-8") as fp:
                fp.write(self.content)
                print(f"... wrote {rst_file}")

    def _expand_dataset(
            self,
            dataset: typing.Dict,
    ) -> typing.Dict:
        r"""Expand dataset dict by additional entries.

        Additional properties are added
        that are only part of the data card,
        but not the dataset object,
        e.g. :meth:`audbcards.Datacard.player`

        Args:
            dataset: dataset object as dictionary representation

        Returns:
            extended datasets dictionary

        """
        # Add audio player for example file
        example = self.example
        dataset['example'] = example
        if example is not None:
            player = self.player(example)
            dataset['player'] = player
        return dataset

    def _render_template(self):
        r"""Render content of data card with Jinja2.

        It uses the dictionary representation
        :attr:`audbcards.Datacard._dataset_dict`
        as bases for rendering.
        The result might vary
        depending if :meth:`audbcards.Datacard._expand_dataset`
        was called before or not.

        """
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
        )
        template = environment.get_template("datacard.j2")

        # Convert dataset object to dictionary
        dataset = self.dataset.properties()

        # Add additional datacard only properties
        dataset = self._expand_dataset(dataset)

        content = template.render(dataset)

        return content
