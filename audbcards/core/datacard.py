import functools
import os
import shutil

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

    """
    def __init__(self, dataset: Dataset):

        self._dataset = dataset

    @functools.cached_property
    def content(self):
        """Property Accessor for rendered jinja2 content."""
        return self._render_template()

    def _render_template(self):

        t_dir = os.path.join(os.path.dirname(__file__), 'templates')
        environment = jinja2.Environment(loader=jinja2.FileSystemLoader(t_dir),
                                         trim_blocks=True)
        template = environment.get_template("datacard.j2")

        # === Add/change content of Dataset class

        # Get dataset property names and content as dictionary
        dataset = self._dataset.properties()

        # Add audio player for example file
        dataset['example'] = self.example
        dataset['player'] = self.player(dataset['example'])

        content = template.render(dataset)

        return content

    @property
    def example(self) -> str:
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
        durations = [
            self._dataset.deps.duration(file)
            for file in self._dataset.deps.media
        ]
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
            media = self._dataset.deps.media[index]
            audb.load_media(
                self._dataset.name,
                media,
                version=self._dataset.version,
                cache_root=self._dataset.cache_root,
                verbose=False,
            )
        except:  # noqa: E722
            media = ''
        return media

    def player(self, file: str) -> str:
        r"""Create an audio player showing the waveform.

        Args:
            file: input audio file to be used in the player.
                :attr:`audbcards.Datacard.example`
                is a good fit

        """
        # Move file to build folder
        src_dir = (
            f'{self._dataset.cache_root}/'
            f'{audb.flavor_path(self._dataset.name, self._dataset.version)}'
        )
        # NOTE: once we added a sphinx datacard extension
        # to this repository,
        # we can directly use `app.builder.outdir`
        # to get the build dir.
        # https://github.com/audeering/audbcards/issues/2
        build_dir = audeer.path('..', 'build', 'html')
        dst_dir = f'{build_dir}/datasets/{self._dataset.name}'
        audeer.mkdir(os.path.join(dst_dir, os.path.dirname(file)))
        shutil.copy(
            os.path.join(src_dir, file),
            os.path.join(dst_dir, file),
        )

        # Add plot of waveform
        signal, sampling_rate = audiofile.read(
            os.path.join(src_dir, file),
            always_2d=True,
        )
        plt.figure(figsize=[3, .5])
        ax = plt.subplot(111)
        audplot.waveform(signal[0, :], ax=ax)
        set_plot_margins()
        plt.savefig(f'{self._dataset.name}.png')
        plt.close()

        player_src = f'{self._dataset.name}/{file}'
        player_str = (
            f'.. image:: ../{self._dataset.name}.png\n'
            '\n'
            '.. raw:: html\n'
            '\n'
            f'    <p><audio controls src="{player_src}"></audio></p>'
        )
        return player_str

    def save(self, ofpath: str = None):
        """Save content of rendered template to rst.

        Args:
            ofpath: filepath to save rendered template to
        Returns:
            None

        if ofpath is specified, the directory must exist.
        """
        if ofpath is None:
            ofpath = f'datasets/{self._dataset.name}.rst'

        with open(ofpath, mode="w", encoding="utf-8") as fp:
            fp.write(self.content)
            print(f"... wrote {ofpath}")
