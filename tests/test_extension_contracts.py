"""Lazy frontend extension lookup preserves canonical module identity."""

import importlib
from importlib.metadata import EntryPoint
from types import ModuleType, SimpleNamespace
import sys
import pytest
from ohagi.compat import plugins, aliases


@pytest.fixture
def registries(monkeypatch):
    monkeypatch.setattr(plugins, "_symbols", {})
    monkeypatch.setattr(aliases._FINDER, "aliases", {})
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))


def test_registered_and_entrypoint_symbols(monkeypatch, registries):
    module = ModuleType("ohagi_test_namespace")
    module.__all__ = []
    monkeypatch.setitem(sys.modules, module.__name__, module)
    entries = (
        EntryPoint(
            name="ohagi_test_namespace.pi", value="math:pi", group="ohagi.symbols"
        ),
    )
    monkeypatch.setattr(
        plugins, "_entries", lambda group: entries if group == "ohagi.symbols" else ()
    )
    plugins.register_symbols(
        {
            "ohagi_test_namespace.sqrt": "math:sqrt",
            "ohagi_test_namespace.nested.value": "math:e",
        }
    )
    assert module.__all__ == ["sqrt"]
    assert plugins.public_symbols(module.__name__) == ["pi", "sqrt"]
    assert plugins.public_symbol(module.__name__, "sqrt")(9) == 3
    assert (
        plugins.public_symbol(module.__name__, "pi")
        == importlib.import_module("math").pi
    )
    with pytest.raises(AttributeError, match="absent"):
        plugins.public_symbol(module.__name__, "absent")
    plugins.register_symbols({"unloaded.value": "math:e"})


def test_alias_namespace_conflicts_and_canonical_identity(monkeypatch, registries):
    finder = aliases._FINDER
    aliases.install_aliases(
        {"ohagi.testing_alias.child": ("math", False), "math": ("math", False)}
    )
    assert aliases.registered_aliases()["ohagi.testing_alias.child"] == ("math", False)
    assert finder.find_spec("ohagi.testing_alias").submodule_search_locations == []
    assert finder.find_spec("unrelated") is None
    spec = finder.find_spec("ohagi.testing_alias.child")
    module = spec.loader.create_module(spec)
    spec.loader.exec_module(module)
    assert module is importlib.import_module("math")
    with pytest.raises(ImportError, match="Conflicting"):
        aliases.install_aliases({"ohagi.testing_alias.child": ("json", False)})
    monkeypatch.setitem(sys.modules, "ohagi.already_loaded", ModuleType("other"))
    with pytest.raises(ImportError, match="already loaded"):
        aliases.install_aliases({"ohagi.already_loaded": ("math", False)})
    namespace = {}
    aliases.reexport_module(namespace, "math")
    assert namespace["sqrt"](16) == 4
    assert namespace["__getattr__"]("pi") == module.pi
    assert "sqrt" in namespace["__dir__"]()


def test_lazy_plugin_namespace_and_busy_guard(monkeypatch, registries):
    finder = plugins._Plugins()
    assert finder.find_spec("unrelated") is None
    finder.busy = True
    assert finder.find_spec("ohagi.testing_alias") is None
    finder.busy = False
    entry = SimpleNamespace(
        name="ohagi.testing_alias.child",
        load=lambda: {"ohagi.testing_alias.child": ("math", False)},
    )
    monkeypatch.setattr(
        plugins, "_entries", lambda group: (entry,) if group == "ohagi.compat" else ()
    )
    namespace = finder.find_spec("ohagi.testing_alias")
    assert namespace.loader.create_module(namespace) is None
    namespace.loader.exec_module(ModuleType("ohagi.testing_alias"))
    spec = finder.find_spec("ohagi.testing_alias.child")
    assert spec.loader.create_module(spec) is importlib.import_module("math")
    assert not finder.busy
    assert finder.find_spec("ohagi.unregistered_namespace") is None
    monkeypatch.setattr(
        plugins,
        "_entries",
        lambda group: (
            SimpleNamespace(
                name="ohagi.failure",
                load=lambda: (_ for _ in ()).throw(RuntimeError("bad extension")),
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="bad extension"):
        finder.find_spec("ohagi.failure")
    assert not finder.busy
