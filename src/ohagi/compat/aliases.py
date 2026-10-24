"""Lazy aliases point to canonical OHAGI or PURIN modules.

Loading the same Python file under two names duplicates classes and registries.
This loader instead returns the already imported canonical module object. Exact
alias catalogs prevent unrelated packages from being intercepted.
"""

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
from importlib.machinery import PathFinder
from threading import RLock


class _AliasLoader(Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        """Create the compatibility namespace requested by the import machinery."""
        return import_module(self.target)

    def exec_module(self, module):
        """Populate an imported compatibility namespace with its registered resources."""
        pass


class _AliasFinder(MetaPathFinder):
    def __init__(self):
        self.aliases = {}
        self.lock = RLock()

    def find_spec(self, fullname, path=None, target=None):
        """Resolve a registered compatibility namespace during module import."""
        entry = self.aliases.get(fullname)
        if entry is None:
            if (
                any(name.startswith(fullname + ".") for name in self.aliases)
                and PathFinder.find_spec(fullname, path) is None
            ):
                from .plugins import _Namespace

                return spec_from_loader(fullname, _Namespace(), is_package=True)
            return None
        destination, is_package = entry
        return spec_from_loader(
            fullname, _AliasLoader(destination), is_package=is_package
        )


_FINDER = _AliasFinder()


def install_aliases(aliases):
    """Install exact aliases, rejecting mixed old/new modules in one process."""
    with _FINDER.lock:
        for name, entry in aliases.items():
            target, is_package = entry
            if name == target:
                continue
            previous = _FINDER.aliases.get(name)
            if previous is not None and previous != (target, is_package):
                raise ImportError(f"Conflicting alias for {name}")
            module = sys.modules.get(name)
            if module is not None and module.__name__ != target:
                raise ImportError(
                    f"{name} is already loaded from another implementation; use a fresh Python process"
                )
        _FINDER.aliases.update({name: tuple(entry) for name, entry in aliases.items()})
        if _FINDER not in sys.meta_path:
            sys.meta_path.insert(0, _FINDER)


def reexport_module(namespace, target):
    """Preserve public and historical private attributes on a root facade."""
    implementation = import_module(target)
    for name, value in vars(implementation).items():
        if not name.startswith("__") or name in {"__all__", "__version__", "__doc__"}:
            namespace[name] = value
    namespace["__getattr__"] = lambda name: getattr(implementation, name)
    namespace["__dir__"] = lambda: sorted(set(namespace) | set(dir(implementation)))


def registered_aliases():
    """Return the currently registered compatibility module aliases."""
    return dict(_FINDER.aliases)
