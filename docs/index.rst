.. include:: ../README.rst

.. toctree::
    :caption: Getting started
    :hidden:

    install
    sphinx-extension

.. toctree::
    :caption: Example Data Cards
    :maxdepth: 2
    :hidden:

    data-public

.. Warning: the usage of genindex is a hack to get a TOC entry, see
.. https://stackoverflow.com/a/42310803. This might break the usage of sphinx if
.. you want to create something different than HTML output.
.. toctree::
    :caption: API Documentation
    :hidden:

    api/audbcards
    genindex

.. toctree::
    :caption: Development
    :maxdepth: 2
    :hidden:

    contributing
    changelog
