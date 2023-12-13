import os

import sphinx

import audb
import audeer

from audbcards.core.datacard import Datacard
from audbcards.core.dataset import Dataset
from audbcards.core.dataset import create_datasets_page


__version__ = '0.1.0'


# ===== MAIN FUNCTION SPHINX EXTENSION ====================================
def setup(app: sphinx.application.Sphinx):
    r"""Modelcard Sphinx extension."""
    # Config values
    app.add_config_value(
        'audbcards_datasets',
        [
            # folder/name, header, repositories, example
            ('datasets', 'Datasets', [audb.config.REPOSITORIES], True),
        ],
        False,
    )
    app.add_config_value('audbcards_cleanup', True, False)

    # Connect functions to extension
    app.connect('builder-inited', builder_inited)
    app.connect('build-finished', builder_finished)

    return {
        'version': __version__,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }


# ===== SPHINX EXTENSION FUNCTIONS ========================================
#
# All fctions defined here
# are added to the extension
# via app.connect()
# in setup()
#
def builder_inited(app: sphinx.application.Sphinx):
    r"""Emitted when the builder object has been created.

    It is available as ``app.builder``.

    """
    # Read config values
    sections = app.config.audbcards_datasets

    # Gather and build data cards for each requested section
    for (path, header, repositories, example) in sections:

        # Clear existing data cards
        datacard_path = audeer.path(app.srcdir, path)
        audeer.rmdir(datacard_path)
        audeer.mkdir(datacard_path)

        # Restrict available repositories
        current_repos = audb.config.REPOSITORIES
        audb.config.REPOSITORIES = audeer.to_list(repositories)

        print('Get list of available datasets... ', end='', flush=True)
        df = audb.available(only_latest=True)
        df = df.sort_index()
        print('done')

        # Iterate datasets and create data card pages
        names = list(df.index)
        versions = list(df['version'])
        datasets = []
        for (name, version) in zip(names, versions):
            print(f'Parse {name}-{version}... ', end='', flush=True)
            dataset = Dataset(name, version)
            datacard = Datacard(
                dataset,
                path=path,
                sphinx_build_dir=app.builder.outdir,
                sphinx_src_dir=app.srcdir,
                example=example,
            )
            datacard.save()
            datasets.append(dataset)
            print('done')

        # Create datasets overview page
        create_datasets_page(
            datasets,
            audeer.path(app.srcdir, f'{path}.rst'),
            datacards_path=path,
            header=header,
        )

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
    if app.config.audbcards_cleanup:
        sections = app.config.audbcards_datasets
        for (path, _, _, _) in sections:
            datacard_path = audeer.path(app.srcdir, path)
            audeer.rmdir(datacard_path)
            for ext in ['rst', 'csv']:
                file = audeer.path(app.srcdir, f'{path}.{ext}')
                if os.path.exists(file):
                    os.remove(file)
