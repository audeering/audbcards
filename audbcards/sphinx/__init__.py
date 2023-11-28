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
    app.add_config_value('audbcards_output_path', 'datasets', False)
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
    output_path = app.config.audbcards_output_path

    # Clear existing data cards
    datacard_path = audeer.path(app.srcdir, output_path)
    audeer.rmdir(datacard_path)
    audeer.mkdir(datacard_path)

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
        datacard = Datacard(dataset, path=output_path)
        datacard._sphinx_build_dir = app.builder.outdir
        datacard._sphinx_src_dir = app.srcdir
        datacard.save()
        datasets.append(dataset)
        print('done')

    # Create datasets overview page
    create_datasets_page(
        datasets,
        audeer.path(app.srcdir, 'datasets.rst'),
        datacards_path=output_path,
    )


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
    output_path = app.config.audbcards_output_path
    datacard_path = audeer.path(app.srcdir, output_path)
    audeer.rmdir(datacard_path)
