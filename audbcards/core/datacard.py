import functools
import os
import shutil

import jinja2
import matplotlib.pyplot as plt

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
        # Provide Jinja filter access to Python build-ins/functions
        environment.filters.update(
            zip=zip,
            tw=self._trim_trailing_whitespace,
        )
        template = environment.get_template("datacard.j2")

        # Add content not included in Dataset class
        dataset = self._dataset.properties()
        dataset['player'] = self.player(dataset['example'])

        content = template.render(dataset)

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

    def player(self, file: str) -> str:
        r"""Create an audio player showing the waveform.

        Args:
            file: input audio file to be used in the player.
                :attr:`audbcards.Dataset.example`
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
