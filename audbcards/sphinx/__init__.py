from collections.abc import Sequence
import os
import shutil
import warnings

import sphinx
import sphinx.application

import audb
import audeer

from audbcards.core.datacard import Datacard
from audbcards.core.dataset import Dataset
from audbcards.core.dataset import create_datasets_page


__version__ = "0.3.1"
table_preview_css_file = "table-preview.css"
table_preview_js_file = "table-preview.js"

# Supported keys of an entry in the audbcards_datasets config,
# the order is used to convert old tuple entries
section_keys = ["folder", "header", "repositories", "example", "table_preview"]
section_keys_info = f"Supported keys are {section_keys}, where 'folder' is required."
section_tuple_info = (
    "Entries given as tuple or list are deprecated, "
    "but still supported with the values "
    "(folder, header, repositories, example) "
    "or (folder, header, repositories, example, table_preview)."
)


# ===== MAIN FUNCTION SPHINX EXTENSION ====================================
def setup(app: sphinx.application.Sphinx):
    r"""Modelcard Sphinx extension."""
    # Config values
    app.add_config_value(
        "audbcards_datasets",
        [{"folder": "datasets", "header": "Datasets"}],
        False,
    )
    app.add_config_value("audbcards_templates", None, "html")

    # Connect functions to extension
    app.connect("builder-inited", builder_inited)
    app.connect("build-finished", builder_finished)

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


# ===== SPHINX EXTENSION FUNCTIONS ========================================
#
# All functions defined here
# are added to the extension
# via app.connect()
# in setup()
#
def builder_inited(app: sphinx.application.Sphinx):
    r"""Emitted when the builder object has been created.

    It is available as ``app.builder``.

    """
    # Read config values,
    # and validate all entries
    # before deleting or creating any files
    sections = [parse_section(section) for section in app.config.audbcards_datasets]
    datacard_paths = [
        datacard_folder(app.srcdir, section["folder"]) for section in sections
    ]
    template_dir = app.config.audbcards_templates
    if template_dir is not None:
        template_dir = os.path.join(app.srcdir, template_dir)

    # Add CSS and JS files for table preview feature
    static_dir = audeer.mkdir(app.builder.outdir, "_static")
    current_dir = audeer.script_dir()
    for file in [table_preview_css_file, table_preview_js_file]:
        shutil.copyfile(audeer.path(current_dir, file), audeer.path(static_dir, file))
    app.add_css_file(table_preview_css_file)
    app.add_js_file(table_preview_js_file)

    # Gather and build data cards for each requested section
    for section, datacard_path in zip(sections, datacard_paths):
        path = section["folder"]
        header = section["header"]
        repositories = section["repositories"]
        example = section["example"]
        table_preview = section["table_preview"]

        # Clear existing data cards
        audeer.rmdir(datacard_path)
        audeer.mkdir(datacard_path)

        # Restrict available repositories
        current_repos = audb.config.REPOSITORIES
        audb.config.REPOSITORIES = audeer.to_list(repositories)

        try:
            print("Get list of available datasets... ", end="", flush=True)
            df = audb.available(only_latest=True, num_workers=10)
            df = df[~df.index.duplicated(keep="first")]
            df = df.sort_index()
            print("done")

            # Store list of datasets in app
            # to make them accessible in `docs/conf.py`
            app.audbcards = {}
            app.audbcards["df"] = df

            # Iterate datasets and create data card pages
            names = list(df.index)
            versions = list(df["version"])
            datasets = []
            for name, version in zip(names, versions):
                print(f"Parse {name}-{version}... ", end="", flush=True)
                dataset = Dataset(name, version)
                datacard = Datacard(
                    dataset,
                    path=path,
                    sphinx_build_dir=app.builder.outdir,
                    sphinx_src_dir=app.srcdir,
                    template_dir=template_dir,
                    example=example,
                    table_preview=table_preview,
                )
                rst_file = os.path.join(
                    os.path.basename(app.srcdir),
                    path,
                    f"{dataset.name}.rst",
                )
                datacard.save(rst_file)
                datasets.append(dataset)
                out_file = rst_file.replace(
                    str(app.srcdir), os.path.basename(app.srcdir)
                )
                print(f"wrote {out_file}")

            # Create datasets overview page
            create_datasets_page(
                datasets,
                f"{datacard_path}.rst",
                datacards_path=path,
                header=header,
            )
        finally:
            # Restore available repositories,
            # even if building the data cards failed
            audb.config.REPOSITORIES = current_repos


def builder_finished(
    app: sphinx.application.Sphinx,
    exception: sphinx.errors.SphinxError,
):
    r"""Emitted when a build has finished.

    This is emitted,
    before Sphinx exits,
    usually used for cleanup.
    This event is emitted
    even when the build process raised an exception,
    given as the exception argument.
    The exception is reraised in the application
    after the event handlers have run.
    If the build process raised no exception,
    exception will be ``None``.
    This allows to customize cleanup actions
    depending on the exception status.

    """
    # Delete auto-generated data card output folder
    sections = app.config.audbcards_datasets
    for section in sections:
        # Skip invalid entries,
        # as cleanup also runs after an error,
        # e.g. raised for an invalid entry,
        # and should not raise another error
        try:
            datacard_path = datacard_folder(app.srcdir, section_folder(section))
        except (TypeError, ValueError):
            continue
        audeer.rmdir(datacard_path)
        for ext in ["rst", "csv"]:
            file = f"{datacard_path}.{ext}"
            if os.path.exists(file):
                os.remove(file)


# ===== HELPER FUNCTIONS ==================================================
#
def datacard_folder(srcdir: str, folder: str) -> str:
    r"""Path of data card folder of an ``audbcards_datasets`` entry.

    As the data card folder
    is deleted before the data cards are created,
    and after the build has finished,
    it needs to be a sub folder
    of the sphinx source folder.

    Args:
        srcdir: sphinx source folder
        folder: value of the ``"folder"`` key
            of an ``audbcards_datasets`` entry

    Returns:
        absolute path of data card folder

    Raises:
        TypeError: if ``folder`` is not a string
        ValueError: if ``folder`` is not a sub folder
            of ``srcdir``

    """
    if not isinstance(folder, str):
        raise TypeError(
            "The 'folder' of an 'audbcards_datasets' entry "
            f"needs to be a string, but got {type(folder).__name__}: {folder!r}."
        )
    srcdir = audeer.path(srcdir)
    path = audeer.path(srcdir, folder)
    if path == srcdir or os.path.commonpath([srcdir, path]) != srcdir:
        raise ValueError(
            f"The 'folder' {folder!r} of an 'audbcards_datasets' entry "
            f"needs to be a sub folder of the sphinx source folder '{srcdir}'."
        )
    return path


def parse_section(section: dict | Sequence) -> dict:
    r"""Parse entry of the ``audbcards_datasets`` config.

    Args:
        section: entry of the ``audbcards_datasets`` config.
            A dictionary with the keys
            ``"folder"`` (required),
            ``"header"``,
            ``"repositories"``,
            ``"example"``,
            and ``"table_preview"``.
            A tuple with the same entries in that order
            is supported as well,
            where ``"table_preview"`` is optional

    Returns:
        dictionary with all keys,
        missing keys are set to their default values

    Raises:
        TypeError: if ``section`` is not a dictionary or tuple
        ValueError: if ``section`` contains unknown keys,
            misses the ``"folder"`` key,
            or the tuple has not four or five entries

    """
    if isinstance(section, (tuple, list)):
        if len(section) not in [4, 5]:
            raise ValueError(
                "A tuple or list entry in 'audbcards_datasets' needs to have "
                f"four or five values, but {section} has {len(section)}. "
                f"{section_tuple_info} "
                f"Use a dictionary instead. {section_keys_info}"
            )
        warnings.warn(
            "Using tuples as entries of 'audbcards_datasets' is deprecated "
            "and will be removed in a future version. "
            "Use a dictionary instead, e.g. "
            f"{dict(zip(section_keys, section))}.",
            category=FutureWarning,
            stacklevel=2,
        )
        section = dict(zip(section_keys, section))
    elif not isinstance(section, dict):
        raise TypeError(
            "An entry in 'audbcards_datasets' needs to be a dictionary, "
            f"but got {type(section).__name__}: {section!r}. "
            f"{section_keys_info} {section_tuple_info}"
        )

    unknown_keys = [key for key in section if key not in section_keys]
    if unknown_keys:
        raise ValueError(
            f"Unknown key(s) {unknown_keys} in 'audbcards_datasets' entry "
            f"{section}. {section_keys_info}"
        )
    if "folder" not in section:
        raise ValueError(
            f"The 'audbcards_datasets' entry {section} "
            f"needs to have the key 'folder'. {section_keys_info}"
        )

    folder = section["folder"]
    return {
        "folder": folder,
        "header": section.get("header", folder),
        "repositories": section.get("repositories", audb.config.REPOSITORIES),
        "example": section.get("example", True),
        "table_preview": section.get("table_preview", True),
    }


def section_folder(section: dict | Sequence) -> object:
    r"""Folder of entry of the ``audbcards_datasets`` config.

    In contrast to :func:`parse_section`,
    it does not raise an error or warning
    for an invalid entry,
    as it is used during cleanup,
    which runs also after an error.
    The returned value is not validated,
    use :func:`datacard_folder` for that.

    Args:
        section: entry of the ``audbcards_datasets`` config

    Returns:
        value of ``"folder"`` of entry
        or ``None`` if it cannot be determined

    """
    if isinstance(section, dict):
        return section.get("folder")
    if isinstance(section, (tuple, list)) and len(section) > 0:
        return section[0]
    return None
