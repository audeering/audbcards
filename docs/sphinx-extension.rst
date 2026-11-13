Sphinx extension
================

:mod:`audbcards` provides a `sphinx extension`_,
to integrate data cards in a sphinx document.
For listing all available datasets,
you need to include the extension inside
the sphinx configuration file ``docs/conf.py``:

.. code-block:: python

    extensions = [
        # ...
        "audbcards.sphinx",
    ]

This will automatically create data cards
inside ``docs/datasets/``,
the overview page ``docs/datasets.rst``
and the ``docs/datasets.csv`` file
listing all available datasets.
You just need to integrate the overview page
``datasets.rst`` in your other RST files.
E.g. you can list it in a TOC inside ``docs/index.rst``:

.. code-block:: rst

    .. toctree::
        :caption: Datasets
        :maxdepth: 2
        :hidden:

        datasets


Configuration
-------------

You can configure the extension
by the ``audbcards_datasets`` entry
inside ``docs/conf.py``.
It is a list of dictionaries,
each describing a separate list of datasets
and their data cards.
The following keys are supported:

``"folder"``
    Name of the folder
    the data cards are stored in,
    and of the datasets overview page.
    It needs to be a sub folder
    of the sphinx source folder,
    as it is deleted
    before and after building the documentation.
    Each entry needs its own folder.
    This key is required.

``"header"``
    Header of the datasets overview page.
    Defaults to the value of ``"folder"``.

``"repositories"``
    :class:`audb.Repository` or list of repositories
    to look for datasets.
    Defaults to :attr:`audb.config.REPOSITORIES`.

``"example"``
    If ``True``,
    an example section is integrated on the data cards.
    It shows an audio or video player
    for media examples,
    and the content of JSON examples.
    If ``False``,
    the whole example section is omitted.
    Defaults to ``True``.

``"table_preview"``
    If ``True``,
    a preview of the content of each table
    is shown when clicking on a row
    of the tables overview
    of a data card.
    If ``False``,
    the tables overview is still shown,
    but its rows cannot be clicked.
    Defaults to ``True``.

If ``audbcards_datasets`` is not set,
it uses the folder ``datasets``
with the header ``Datasets``.

For example,
to restrict to a single repository
and name the folder
and header to the repository name:

.. code-block:: python

    audbcards_datasets = [
        {
            "folder": "data-public",
            "repositories": audb.Repository(
                name="data-public",
                host="https://audeering.jfrog.io/artifactory",
                backend="artifactory",
            ),
        },
    ]

You would then need to include ``data-public``
in your TOC instead of the default ``datasets`` name.

You can even create several separated lists of datasets
and corresponding data cards,
by providing several entries:

.. code-block:: python

    audbcards_datasets = [
        {
            "folder": "data-public",
            "header": "Public datasets",
            "repositories": audb.Repository(
                name="data-public",
                host="https://audeering.jfrog.io/artifactory",
                backend="artifactory",
            ),
        },
        {
            "folder": "data-private",
            "header": "Private datasets",
            "repositories": audb.Repository(
                name="data-private",
                host="https://audeering.jfrog.io/artifactory",
                backend="artifactory",
            ),
            "example": False,
            "table_preview": False,
        },
    ]

This will produce two distinct RST overview pages,
that can be included in your document,
e.g. in ``docs/index.rst`` you could then write:

.. code-block:: rst

    .. toctree::
        :caption: Datasets
        :maxdepth: 2
        :hidden:

        data-public
        data-private

Entries given as tuples
with the values
``(folder, header, repositories, example)``
or ``(folder, header, repositories, example, table_preview)``
are still supported,
but deprecated
and will be removed in a future version.

A user can also influence
how the resulting datacard appears,
by setting the config the ``audbcards_templates`` folder.
The value needs to be relative to the sphinx source dir
(e.g. ``docs/``).


Referencing
-----------

You can reference dataset overview pages
by their folder name,
e.g.

.. code-block:: rst

    A list of public datasets is shown at :ref:`audb-public`.

Which will render as:

    A list of public datasets is shown at :ref:`audb-public`.

And you can reference single data cards
by a combination of their folder
and dataset name,
e.g.

.. code-block:: rst

    :ref:`audb-public-emodb` shows the data card for emodb.

Which will render as:

    :ref:`audb-public-emodb` shows the data card for emodb.


List of available datasets
--------------------------

The sphinx extension calls :func:`audb.available`
for every entry of ``audbcards_datasets``
to get an overview of all available datasets.
This information can be reused inside ``docs/conf.py``
as it is stored in the ``app.audbcards`` dictionary
under the ``"dfs"`` key,
which maps the ``"folder"`` of each entry
to its list of datasets,
e.g.

.. code-block:: python

    def setup(app: sphinx.application.Sphinx):
        app.connect("builder-inited", list_datasets)


    def list_datasets(app: sphinx.application.Sphinx):
        for folder, df in app.audbcards["dfs"].items():
            # ...

The ``"df"`` key holds the list of datasets
of the last entry of ``audbcards_datasets``.
        

.. _sphinx extension: https://www.sphinx-doc.org/en/master/usage/extensions/index.html
