import functools
import os
import shutil
import typing

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import audb
import audeer
import audiofile
import audplot

from audbcards.core.dataset import Dataset
from audbcards.core.utils import set_plot_margins


class Datacard(object):
    r"""Datacard of a dataset.

    The datacard object
    writes a RST file
    for a given dataset,
    which can then be used
    to generate an HTML datacard page
    using ``sphinx``.

    Args:
        dataset: dataset object
        path: path to folder
            that store datacard files
        example: if ``True``,
            include an audio example in the data card
            showing the waveform of the audio
            and an interactive player
        sphinx_build_dir: build dir of sphinx.
            If not ``None``
            and ``example`` is ``True``,
            a call to :meth:`audbcards.Datacard.player`
            will store an example audio file
            under
            ``<sphinx_build_dir>/<path>/<db-name>/<media-file-in-db>``
        sphinx_src_dir: source dir of sphinx.
            If not ``None``
            and ``example`` is ``True``,
            a call to :meth:`audbcards.Datacard.player`
            will store a wavplot of the example audio file
            under
            ``<sphinx_src_dir>/<path>/<db-name>/<db-name>.png``

    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        path: str = "datasets",
        example: bool = True,
        sphinx_build_dir: str = None,
        sphinx_src_dir: str = None,
    ):
        self.dataset = dataset
        """Dataset object."""

        self.path = path
        """Folder to store datacard."""

        self.example = example
        """If an audio example should be included."""

        self.sphinx_build_dir = sphinx_build_dir
        """Sphinx build dir."""

        self.sphinx_src_dir = sphinx_src_dir
        """Sphinx source dir."""

        self.rst_preamble = ""
        """RST code added at top of data card."""

    @functools.cached_property
    def content(self):
        """Property Accessor for rendered jinja2 content."""
        return self._render_template()

    @property
    def example_media(self) -> typing.Optional[str]:
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
        # Download of example data might fail
        try:
            media = self.dataset.deps.media[index]
            audb.load_media(
                self.dataset.name,
                media,
                version=self.dataset.version,
                verbose=False,
            )
        except:  # noqa: E722
            media = None
        return media

    @property
    def file_duration_distribution(self) -> str:
        r"""Minimum and maximum of files durations, and plotted distribution.

        This generates a single line
        containing the mininimum and maximum values
        of files durations.

        If :attr:`audbcards.Datacard.self.sphinx_src_dir` is set
        (e.g. when used in the sphinx extension),
        an inline image is stored
        in the sphinx source folder
        under ``<dataset-name>/<dataset-name>-file-durations.png``
        and displayed
        between the minimum and maximum values.

        """
        min_ = 0
        max_ = 0
        unit = "s"
        durations = self.dataset.file_durations
        if len(durations) > 0:
            min_ = np.min(durations)
            max_ = np.max(durations)
        distribution_str = f"{min_:.1f} {unit} .. {max_:.1f} {unit}"

        # Save distribution plot
        if self.sphinx_src_dir is not None:
            self._plot_distribution(durations)
            name = "file-durations"
            image_file = audeer.path(
                self.sphinx_src_dir,
                self.path,
                self.dataset.name,
                f"{self.dataset.name}-{name}.png",
            )
            audeer.mkdir(os.path.dirname(image_file))
            plt.savefig(image_file, transparent=True)
            plt.close()
            distribution_str = self._inline_image(
                f"{min_:.1f} {unit}",
                f"./{self.dataset.name}/{self.dataset.name}-{name}.png",
                f"{max_:.1f} {unit}",
            )

        return distribution_str

    def player(
        self,
        file: str = None,
    ) -> str:
        r"""Create an audio player showing the waveform.

        Args:
            file: input audio file to be used in the player.
                If ``None``,
                :attr:`audbcards.Datacard.example_media`
                is used

        """
        if file is None:
            file = self.example_media

        # use audb cache instead of dataset.cache_root
        media_src_dir = (
            f"{audb.default_cache_root()}/"
            f"{audb.flavor_path(self.dataset.name, self.dataset.version)}"
        )

        # Move file to build folder
        if self.sphinx_build_dir is not None:
            media_dst_dir = audeer.path(
                self.sphinx_build_dir,
                self.path,
                self.dataset.name,
            )
            audeer.mkdir(os.path.join(media_dst_dir, os.path.dirname(file)))
            shutil.copy(
                os.path.join(media_src_dir, file),
                os.path.join(media_dst_dir, file),
            )

        # Add plot of waveform
        if self.sphinx_src_dir is not None:
            signal, sampling_rate = audiofile.read(
                os.path.join(media_src_dir, file),
                always_2d=True,
            )
            image_file = audeer.path(
                self.sphinx_src_dir,
                self.path,
                self.dataset.name,
                f"{self.dataset.name}.png",
            )
            audeer.mkdir(os.path.dirname(image_file))
            plt.figure(figsize=[3, 0.5])
            ax = plt.subplot(111)
            audplot.waveform(signal[0, :], ax=ax)
            set_plot_margins()
            plt.savefig(image_file)
            plt.close()

        player_src = f"./{self.dataset.name}/{file}"
        player_str = (
            f".. image:: ./{self.dataset.name}/{self.dataset.name}.png\n"
            "\n"
            ".. raw:: html\n"
            "\n"
            f'    <p><audio controls src="{player_src}"></audio></p>'
        )
        return player_str

    def save(self, file: str = None):
        """Save content of rendered template to rst.

        Args:
            file: name of output RST file.
                If ``None``
                and :attr:`audbcards.Datacard.sphinx_src_dir`
                is not ``None``,
                the RST file will be stored
                as ``<sphinx_src_dir>/<path>/<dataset>.rst``

        """
        if file is None and self.sphinx_src_dir is not None:
            file = audeer.path(
                self.sphinx_src_dir,
                self.path,
                f"{self.dataset.name}.rst",
            )
        if file is not None:
            with open(file, mode="w", encoding="utf-8") as fp:
                fp.write(self.content)

    def _inline_image(
        self,
        text1: str,
        file: str,
        text2: str,
    ) -> str:
        r"""RST string for rendering inline image between text.

        Args:
            text1: text to the left of the image
            file: image file
            text2: text to the right of the image

        Returns:
            RST code to generate the desired inline image

        """
        # In RST there is no easy way to insert inline images.
        # We use the following workaround:
        #
        # .. |ref| image:: file
        #
        # text1 |ref| text2
        #
        ref = audeer.basename_wo_ext(file)
        self.rst_preamble += f".. |{ref}| image:: {file}\n"
        return f"{text1} |{ref}| {text2}"

    def _plot_distribution(
        self,
        values: typing.Sequence,
    ):
        r"""Plot inline distribution.

        Args:
            values: sequence of values

        """
        if len(values) == 0:
            min_ = 0
            max_ = 0
        else:
            min_ = np.min(values)
            max_ = np.max(values)
        plt.figure(figsize=[0.5, 0.15])
        # Remove all margins besides bottom
        plt.subplot(111)
        plt.subplots_adjust(
            left=0,
            bottom=0.25,
            right=1,
            top=1,
            wspace=0,
            hspace=0,
        )
        # Plot duration distribution
        sns.kdeplot(
            values,
            fill=True,
            cut=0,
            clip=(min_, max_),
            linewidth=0,
            alpha=1,
            color="#d54239",
        )
        # Remove all tiks, labels
        sns.despine(left=True, bottom=True)
        plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        plt.xlabel("")
        plt.ylabel("")

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
        # Add path of datacard folder
        dataset["path"] = self.path
        # Add audio player for example file
        dataset["example"] = None
        if self.example:
            example = self.example_media
            if example is not None:
                player = self.player(example)
                dataset["player"] = player
                dataset["example"] = example
        dataset["file_duration_distribution"] = self.file_duration_distribution
        return dataset

    def _render_template(self) -> str:
        r"""Render content of data card with Jinja2.

        It uses the dictionary representation
        :attr:`audbcards.Datacard._dataset_dict`
        as bases for rendering.
        The result might vary
        depending if :meth:`audbcards.Datacard._expand_dataset`
        was called before or not.

        """
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
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

        # Add RST preamble
        if len(self.rst_preamble) > 0:
            content = self.rst_preamble + "\n" + content

        return content
