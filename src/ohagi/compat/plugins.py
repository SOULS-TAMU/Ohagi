"""Lazy plugin discovery; the core has no knowledge of frontend packages."""

from importlib.metadata import entry_points
from functools import lru_cache
from importlib import import_module
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import PathFinder
from importlib.util import spec_from_loader
import sys


@lru_cache(maxsize=None)
def _entries(group):
    return tuple(entry_points(group=group))


_symbols = {}


def public_symbols(module):
    """Return the public symbols registered for a frontend namespace."""
    prefix = module + "."
    names = set(_symbols)
    names.update(ep.name for ep in _entries("ohagi.symbols"))
    return sorted(
        key[len(prefix) :]
        for key in names
        if key.startswith(prefix) and "." not in key[len(prefix) :]
    )


def register_symbols(values):
    """Register frontend symbols for compatibility lookup."""
    _symbols.update(values)
    for key in values:
        parent, name = key.rsplit(".", 1)
        module = sys.modules.get(parent)
        exports = getattr(module, "__all__", None) if module is not None else None
        if isinstance(exports, list) and name not in exports:
            exports.append(name)


def public_symbol(module, name):
    """Resolve a named public symbol from the compatibility registry."""
    key = module + "." + name
    destination = _symbols.get(key)
    if destination is None:
        matches = tuple(ep for ep in _entries("ohagi.symbols") if ep.name == key)
        if not matches:
            raise AttributeError(name)
        destination = matches[0].value
    owner, symbol = destination.split(":", 1)
    return getattr(import_module(owner), symbol)


class _Namespace(Loader):
    def create_module(self, spec):
        """Create the compatibility namespace requested by the import machinery."""
        return None

    def exec_module(self, module):
        """Populate an imported compatibility namespace with its registered resources."""
        pass


class _Plugins(MetaPathFinder):
    busy = False

    def find_spec(self, fullname, path=None, target=None):
        """Resolve a registered compatibility namespace during module import."""
        if self.busy or not fullname.startswith("ohagi."):
            return None
        if PathFinder.find_spec(fullname, path) is not None:
            return None
        matches = tuple(
            ep
            for ep in _entries("ohagi.compat")
            if fullname == ep.name or fullname.startswith(ep.name + ".")
        )
        if matches:
            self.busy = True
            try:
                from .aliases import install_aliases, _FINDER

                install_aliases(matches[0].load())
                return _FINDER.find_spec(fullname, path, target)
            finally:
                self.busy = False
        if any(ep.name.startswith(fullname + ".") for ep in _entries("ohagi.compat")):
            return spec_from_loader(fullname, _Namespace(), is_package=True)
        return None


sys.meta_path.append(_Plugins())
