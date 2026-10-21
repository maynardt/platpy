"""
lazy_value.py -- a generic "don't load until touched" wrapper.

Different problem than lazy_transcripts.py: that stays dask/chunk-lazy even
after resolution, because partial reads of a multi-million-row point cloud
are the whole point. This is for values where there's no partial-read
benefit once something actually needs them (a cell boundary vertex table, a
cached cell x gene matrix) -- the only win available is not paying for the
read/deserialize at all when nothing ever asks. Resolves once, caches the
result, and releases the loader closure afterward.
"""
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Deferred:
    loader: Any  # Callable[[], Any] -- not type-hinted as Callable to keep this import-light
    _resolved: Any = field(default=None, repr=False)
    _done: bool = field(default=False, repr=False)

    def resolve(self) -> Any:
        if not self._done:
            self._resolved = self.loader()
            self._done = True
            self.loader = None  # release the closure (and whatever it closed over) once unneeded
        return self._resolved


def resolve(value: Any) -> Any:
    """Resolve value if it's a Deferred, otherwise return it unchanged."""
    return value.resolve() if isinstance(value, Deferred) else value
