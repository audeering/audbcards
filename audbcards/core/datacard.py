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

from audbcards.core.config import config
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
            include an audio or video example in the data card
            showing the waveform of the audio
            and an interactive player
        sphinx_build_dir: build dir of sphinx.
            If not ``None``
            and ``example`` is ``True``,
            a call to :meth:`audbcards.Datacard.player`
            will store an example audio file
            under
            ``<sphinx_build_dir>/<path>/<dataset-name>/``
        sphinx_src_dir: source dir of sphinx.
            If not ``None``
            and ``example`` is ``True``,
            a call to :meth:`audbcards.Datacard.player`
            will store a waveform plot of the example audio file
            under
            ``<sphinx_src_dir>/<path>/<dataset-name>/``
        cache_root: cache folder.
            If ``None``,
            the environmental variable ``AUDBCARDS_CACHE_ROOT``,
            or :attr:`audbcards.config.CACHE_ROOT`
            is used

    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        path: str = "datasets",
        example: bool = True,
        sphinx_build_dir: str = None,
        sphinx_src_dir: str = None,
        cache_root: str = None,
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

        if cache_root is None:
            cache_root = os.environ.get("AUDBCARDS_CACHE_ROOT") or config.CACHE_ROOT
        self.cache_root = audeer.mkdir(cache_root)
        r"""Cache root folder."""

        self.rst_preamble = ""
        """RST code added at top of data card."""

    @functools.cached_property
    def content(self):
        """Property Accessor for rendered jinja2 content."""
        return self._render_template()

    @property
    def file_duration_distribution(self) -> str:
        r"""Minimum and maximum of files durations, and plotted distribution.

        This generates a single line
        containing the mininimum and maximum values
        of files durations.

        If :attr:`audbcards.Datacard.sphinx_src_dir` is not ``None``
        (e.g. when used in the sphinx extension),
        and the dataset contains audio or video files,
        an image is stored in the file
        ``<dataset-name>-<dataset-version>-file-duration-distribution.png``,
        which is cached in
        ``<cache-root>/<dataset-name>/<dataset-version>/``
        and copied to the sphinx source folder
        into
        ``<sphinx-src-dir>/<path><dataset-name>/``.
        The image is displayed inline
        between the minimum and maximum values.
        If all duration values are the same,
        no distribution plot is created.

        """
        file_name = (
            f"{self.dataset.name}-{self.dataset.version}-file-duration-distribution.png"
        )
        # Cache is organized as `<cache_root>/<name>/<version>/`
        cache_file = audeer.path(
            self.cache_root,
            self.dataset.name,
            self.dataset.version,
            file_name,
        )

        min_ = 0
        max_ = 0
        unit = "s"
        durations = self.dataset.file_durations
        if len(durations) > 0:
            min_ = np.min(durations)
            max_ = np.max(durations)

        # Skip creating a distribution plot,
        # if all durations are the same
        if min_ == max_:
            return f"each file is {max_:.1f} {unit}"

        distribution_str = f"{min_:.1f} {unit} .. {max_:.1f} {unit}"

        # Save distribution plot
        if self.sphinx_src_dir is not None:
            # Plot distribution to cache,
            # if not found there already.
            if not os.path.exists(cache_file):
                audeer.mkdir(os.path.dirname(cache_file))
                self._plot_distribution(durations)
                plt.savefig(cache_file, transparent=True)
                plt.close()

            image_file = audeer.path(
                self.sphinx_src_dir,
                self.path,
                self.dataset.name,
                file_name,
            )
            audeer.mkdir(os.path.dirname(image_file))
            shutil.copyfile(cache_file, image_file)
            distribution_str = self._inline_image(
                f"{min_:.1f} {unit}",
                f"./{self.dataset.name}/{file_name}",
                f"{max_:.1f} {unit}",
            )

        return distribution_str

    def player(self) -> str:
        r"""Create an audio/video player showing the waveform.

        If :attr:`audbcards.Datacard.sphinx_build_dir`
        or :attr:`audbcards.Datacard.sphinx_src_dir`
        is not ``None``,
        an example media file is cached in the folder
        ``<dataset-name>-<dataset-version>-player-media/``
        inside
        ``<cache-root>/<dataset-name>/<dataset-version>/``,
        using the same sub-folder structure
        as the media file has inside its dataset.
        If :attr:`audbcards.Datacard.sphinx_build_dir`
        is not ``None``,
        the media sub-folder structure
        is also copied
        to the sphinx build dir into
        ``<sphinx-build-dir>/<path>/<dataset-name>/``,
        and an audio element referencing this file
        is added to the returned RST string.

        If :attr:`audbcards.Datacard.sphinx_src_dir` is not ``None``,
        a plot of the waveform of the media file
        is cached under
        ``<dataset-name>-<dataset-version>-player-waveform.png``
        inside
        ``<cache-root>/<dataset-name>/<dataset-version>/``.
        It is also copied to the sphinx source folder into
        ``<sphinx-src-dir>/<path>/<dataset-name>/``,
        and referenced at the beginning of the returned RST string.

        If :attr:`audbcards.Datacard.sphinx_build_dir`
        and :attr:`audbcards.Datacard.sphinx_src_dir`
        are ``None``,
        an empty string is returned.

        Returns:
            String containing RST code to include the player

        """
        # Cache is organized as `<cache_root>/<name>/<version>/`
        cache_folder = audeer.path(
            self.cache_root,
            self.dataset.name,
            self.dataset.version,
        )

        def load_media_to_cache() -> str:
            r"""Load media file with audb and copy to audbcards cache.

            Load example media file to cache,
            if not existent.

            Returns:
                full path to media file in cache

            """
            cache_example_media = audeer.path(
                cache_folder,
                f"{self.dataset.name}-{self.dataset.version}-player-media",
                self.dataset.example_media,
            )
            if not os.path.exists(cache_example_media):
                media = audb.load_media(
                    self.dataset.name,
                    self.dataset.example_media,
                    version=self.dataset.version,
                    verbose=False,
                )[0]
                audeer.mkdir(os.path.dirname(cache_example_media))
                shutil.copy(media, cache_example_media)
            return cache_example_media

        def plot_waveform_to_cache(cache_example_media: str) -> str:
            r"""Plot waveform of example media to cache.

            Args:
                cache_example_media: full path to media file in cache

            Returns:
                full path to waveform file in cache

            """
            cache_waveform_file = audeer.path(
                cache_folder,
                f"{self.dataset.name}-{self.dataset.version}-player-waveform.png",
            )
            if not os.path.exists(cache_waveform_file):
                signal, sampling_rate = audiofile.read(
                    cache_example_media,
                    always_2d=True,
                )
                audeer.mkdir(os.path.dirname(cache_waveform_file))
                plt.figure(figsize=[3, 0.5])
                ax = plt.subplot(111)
                audplot.waveform(signal[0, :], ax=ax)
                set_plot_margins()
                plt.savefig(cache_waveform_file)
                plt.close()
            return cache_waveform_file

        # String holding the RST code to include the player
        player_str = ""

        # Add plot of waveform to Sphinx source folder (e.g. docs/)
        if self.sphinx_src_dir is not None:
            cache_example_media = load_media_to_cache()
            cache_waveform_file = plot_waveform_to_cache(cache_example_media)
            plot_dst_dir = audeer.path(
                self.sphinx_src_dir,
                self.path,
                self.dataset.name,
            )
            audeer.mkdir(plot_dst_dir)
            shutil.copy(
                cache_waveform_file,
                os.path.join(plot_dst_dir, os.path.basename(cache_waveform_file)),
            )
            waveform_src = (
                f"./{self.dataset.name}/{os.path.basename(cache_waveform_file)}"
            )
            player_str += f".. image:: {waveform_src}\n\n"

        # Copy media file to Sphinx build folder (e.g. build/)
        if self.sphinx_build_dir is not None:
            cache_example_media = load_media_to_cache()
            media_dst_dir = audeer.path(
                self.sphinx_build_dir,
                self.path,
                self.dataset.name,
            )
            audeer.mkdir(media_dst_dir, os.path.dirname(self.dataset.example_media))
            shutil.copy(
                cache_example_media,
                os.path.join(media_dst_dir, self.dataset.example_media),
            )
            if audiofile.has_video(cache_example_media):
                media_tag = "video"
            else:
                media_tag = "audio"
            player_src = f"./{self.dataset.name}/{self.dataset.example_media}"
            player_str += (
                ".. raw:: html\n"
                "\n"
                f'    <p><{media_tag} controls src="{player_src}"></{media_tag}></p>'
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

    @property
    def segment_duration_distribution(self) -> str:
        r"""Minimum and maximum of segment durations, and plotted distribution.

        This generates a single line
        containing the mininimum and maximum values
        of segment durations.

        If :attr:`audbcards.Datacard.sphinx_src_dir` is not ``None``
        (e.g. when used in the sphinx extension),
        and the dataset contains segments,
        an image is stored in the file
        ``<dataset-name>-<dataset-version>-segment-duration-distribution.png``,
        which is cached in
        ``<cache-root>/<dataset-name>/<dataset-version>/``
        and copied to the sphinx source folder
        into
        ``<sphinx-src-dir>/<path><dataset-name>/``.
        The image is displayed inline
        between the minimum and maximum values.
        If all duration values are the same,
        no distribution plot is created.

        """
        file_name = (
            f"{self.dataset.name}-{self.dataset.version}"
            "-segment-duration-distribution.png"
        )
        # Cache is organized as `<cache_root>/<name>/<version>/`
        cache_file = audeer.path(
            self.cache_root,
            self.dataset.name,
            self.dataset.version,
            file_name,
        )

        min_ = 0
        max_ = 0
        unit = "s"
        durations = self.dataset.segment_durations
        if len(durations) > 0:
            min_ = np.min(durations)
            max_ = np.max(durations)

        # Skip creating a distribution plot,
        # if all durations are the same
        if min_ == max_:
            return f"each file is {max_:.1f} {unit}"

        distribution_str = f"{min_:.1f} {unit} .. {max_:.1f} {unit}"

        # Save distribution plot
        if self.sphinx_src_dir is not None:
            # Plot distribution to cache,
            # if not found there already.
            if not os.path.exists(cache_file):
                audeer.mkdir(os.path.dirname(cache_file))
                self._plot_distribution(durations)
                plt.savefig(cache_file, transparent=True)
                plt.close()

            image_file = audeer.path(
                self.sphinx_src_dir,
                self.path,
                self.dataset.name,
                file_name,
            )
            audeer.mkdir(os.path.dirname(image_file))
            shutil.copyfile(cache_file, image_file)
            distribution_str = self._inline_image(
                f"{min_:.1f} {unit}",
                f"./{self.dataset.name}/{file_name}",
                f"{max_:.1f} {unit}",
            )

        return distribution_str

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
        if self.example:
            if self.dataset.example_media is not None:
                player = self.player()
                dataset["player"] = player
        dataset["file_duration_distribution"] = self.file_duration_distribution
        dataset["segment_duration_distribution"] = self.segment_duration_distribution
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
        dataset = self.dataset._cached_properties()

        # Add additional datacard only properties
        dataset = self._expand_dataset(dataset)

        content = template.render(dataset)

        # Add RST preamble
        if len(self.rst_preamble) > 0:
            content = self.rst_preamble + "\n" + content

        return content
