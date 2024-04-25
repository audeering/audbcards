class config:
    r"""Get/set configuration values for the :mod:`audbcards` module.

    Examples:
        >>> config.CACHE_ROOT
        '~/.cache/audbcards'
        >>> config.CACHE_ROOT = "~/audbcards"
        >>> config.CACHE_ROOT
        '~/audbcards'

    .. Ensure to restore the default value
        >>> config.CACHE_ROOT = "~/.cache/audbcards"

    """

    CACHE_ROOT = "~/.cache/audbcards"
    r"""Default cache folder.

    It can be overwritten
    by the ``AUDBCARDS_CACHE_ROOT`` environment variable.

    """
