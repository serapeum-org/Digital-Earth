"""Tests for RP.11 — plugin discovery via entry points (digitalearth.plugins)."""

from importlib.metadata import EntryPoint

import pytest

from digitalearth.plugins import GROUPS, iter_plugins, load_plugins


class _FakeEP:
    """A minimal EntryPoint stand-in: a name plus a load() returning a fixed object."""

    def __init__(self, name, target):
        """Store the entry-point name and the object its load() should return."""
        self.name = name
        self._target = target

    def load(self):
        """Return the pre-set target object (stands in for importing the entry point)."""
        return self._target


class TestGroups:
    """Tests for the GROUPS constant."""

    def test_declares_known_extension_points(self):
        """GROUPS advertises the styles and sources extension points."""
        assert "digitalearth.styles" in GROUPS and "digitalearth.sources" in GROUPS


class TestIterPlugins:
    """Tests for iter_plugins."""

    def test_unknown_group_is_empty(self):
        """Querying an unused group from the real environment yields nothing."""
        assert list(iter_plugins("digitalearth.nonexistent")) == []

    def test_yields_supplied_entry_points_without_loading(self):
        """With eps supplied, the entry points are yielded verbatim (not loaded)."""
        eps = [EntryPoint("demo", "pkg.mod:OBJ", "digitalearth.styles")]
        names = [e.name for e in iter_plugins("digitalearth.styles", eps=eps)]
        assert names == ["demo"], f"expected ['demo'], got {names}"


class TestLoadPlugins:
    """Tests for load_plugins."""

    def test_empty_when_nothing_registered(self):
        """An unused group loads to an empty mapping."""
        assert load_plugins("digitalearth.nonexistent") == {}

    def test_loads_targets_by_name(self):
        """Each supplied entry point is loaded and keyed by its name."""
        eps = [_FakeEP("extra", {"cmap": "magma"}), _FakeEP("more", [1, 2])]
        loaded = load_plugins("digitalearth.styles", eps=eps)
        assert loaded == {"extra": {"cmap": "magma"}, "more": [1, 2]}, f"unexpected: {loaded}"
