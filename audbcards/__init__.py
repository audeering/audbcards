from audbcards.core.config import config
from audbcards.core.datacard import Datacard
from audbcards.core.dataset import Dataset


__all__ = []


# Dynamically get the version of the installed module
try:
    import importlib.metadata

    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    importlib = None  # pragma: no cover
finally:
    del importlib
