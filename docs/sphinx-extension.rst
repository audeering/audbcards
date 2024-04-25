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
to use only certain :class:`audb.Repository`
to look for datasets,
select a name for the folder
the data cards are stored,
and select if an audio example should be integrated
on the data cards.
All of those options are handled
by the ``audbcards_datasets`` entry
inside ``docs/conf.py``.
For example,
to restrict to a single repository
and name the folder
and header to the repository name:

.. code-block:: python

    audbcards_datasets = [
        (   
            "data-public",  # folder name
            "data-public",  # datasets overview page header
            audb.Repository(  # repository to use, can be a list of repos
                name="data-public",
                host="https://audeering.jfrog.io/artifactory",
                backend="artifactory",
            ),
            True,  # show audio examples
        ),  
    ]

You would then need to include ``data-public``
in your TOC instead of the default ``datasets`` name.

You can even create several separated lists of datasets
and corresponding data cards.
In ``docs/conf.py`` you provide a list of entries:

.. code-block:: python

    audbcards_datasets = [
        (   
            "data-public",  # folder name
            "data-public",  # datasets overview page header
            audb.Repository(  # repository to use, can be a list of repos
                name="data-public",
                host="https://audeering.jfrog.io/artifactory",
                backend="artifactory",
            ),
            True,  # show audio examples
        ),  
        (   
            "data-private",  # folder name
            "data-private",  # datasets overview page header
            audb.Repository(  # repository to use, can be a list of repos
                name="data-private",
                host="https://audeering.jfrog.io/artifactory",
                backend="artifactory",
            ),
            False,  # don't show audio examples
        ),  
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


Referencing
-----------

You can reference dataset overview pages
by their folder name,
e.g.

.. code-block:: rst

    A list of public datasets is shown at :ref:`data-public`.

Which will render as:

    A list of public datasets is shown at :ref:`data-public`.

And you can reference single data cards
by a combination of their folder
and dataset name,
e.g.

.. code-block:: rst

    :ref:`data-public-emodb` shows the data card for emodb.

Which will render as:

    :ref:`data-public-emodb` shows the data card for emodb.


List of available datasets
--------------------------

The sphinx extension calls :func:`audb.available`
to get an overview of all available datasets.
This information can be reused inside ``docs/conf.py``
as it is stored in the ``app.audbcards`` dictionary
under the ``"df"`` key, e.g.

.. code-block:: python

    def setup(app: sphinx.application.Sphinx):
        df = app.audbcards["df"]
        # ...
        

.. _sphinx extension: https://www.sphinx-doc.org/en/master/usage/extensions/index.html
