"""`DataRef` and the resolver registry — a layer refers to data instead of holding it (DE-14, #273).

Holding the data object inside the layer is what blocks dynamic tiling, level-of-detail and any figure that
wants to round-trip through a dict. These cover the reference and the registry that opens it.
"""

from pathlib import Path

import pytest

from digitalearth.base.registry import (
    SOURCES_GROUP,
    clear_objects,
    register_object,
    register_resolver,
    resolve_uri,
    resolvers,
)
from digitalearth.base.spec import DataRef


@pytest.fixture(autouse=True)
def _clean_object_table():
    """Drop registered in-memory objects between cases, so ids cannot leak across tests."""
    yield
    clear_objects()


class TestTheReference:
    """A reference is a value: comparable, serialisable, and refuses to name nothing."""

    def test_a_path_is_all_a_reference_needs(self):
        """The common case is one field.

        Test scenario:
            A stored figure should not fill with nulls for hints nobody set.
        """
        assert DataRef("data/dem.tif").to_dict() == {"uri": "data/dem.tif"}, (
            "unset hints must be omitted"
        )

    def test_it_round_trips_through_a_dict(self):
        """`to_dict` and `from_dict` are inverses.

        Test scenario:
            This is the property that makes a figure serialisable at all — the reason a layer must not hold a
            live Dataset.
        """
        ref = DataRef("s3://bucket/x.tif", driver="COG", version="2024-01")
        assert DataRef.from_dict(ref.to_dict()) == ref, (
            "a reference must survive a round trip"
        )

    def test_an_empty_uri_is_refused(self):
        """A reference naming nothing resolves to whatever the working directory is.

        Test scenario:
            That failure only shows on another machine, which is the worst time to find it.
        """
        with pytest.raises(ValueError, match="non-empty uri"):
            DataRef("   ")

    def test_an_unknown_key_is_refused_rather_than_dropped(self):
        """A dict carrying a field this version does not know raises.

        Test scenario:
            Silently dropping it would lose data a newer writer meant to keep, and the figure would come back
            subtly different from what was saved.
        """
        with pytest.raises(ValueError, match="unknown keys"):
            DataRef.from_dict({"uri": "a.tif", "bogus": 1})

    def test_it_cannot_be_mutated(self):
        """Frozen, like the rest of the vocabulary.

        Test scenario:
            Two layers may share one reference; one editing it would move the other's data.
        """
        ref = DataRef("a.tif")
        with pytest.raises(AttributeError):
            ref.uri = "b.tif"


class TestTheRegistry:
    """Turning a name into data, without `base/` knowing any reader."""

    def test_the_builtin_schemes_are_registered(self):
        """`file` and `object` resolve out of the box.

        Test scenario:
            The default file resolver is what makes a plain path work with no setup.
        """
        assert {"file", "object"} <= set(resolvers()), (
            f"the built-in schemes must be registered, got {sorted(resolvers())}"
        )

    def test_a_third_party_resolver_can_be_registered(self):
        """A new scheme becomes resolvable without touching this package.

        Test scenario:
            The open-registry property. A plugin declares itself under the entry-point group rather than
            editing a table here.
        """
        register_resolver("demo-scheme", lambda uri: f"opened:{uri}")
        assert DataRef("demo-scheme://x").open() == "opened:demo-scheme://x", (
            "a registered resolver must be reached"
        )

    def test_an_unknown_scheme_lists_the_known_ones(self):
        """The error distinguishes a typo from a plugin that failed to install.

        Test scenario:
            Both look identical otherwise, and the second is the one a user cannot debug.
        """
        with pytest.raises(KeyError, match="known schemes are"):
            DataRef("nosuchscheme://x").open()

    def test_the_entry_point_group_is_the_shared_contract_string(self):
        """The registry advertises the same group the plugin loader already discovers.

        Test scenario:
            A second group name would mean a plugin could install correctly and still never be found.
        """
        from digitalearth.ops.plugins import GROUPS

        assert SOURCES_GROUP in GROUPS, (
            f"{SOURCES_GROUP!r} must be one of the groups the plugin loader reads, got {GROUPS}"
        )

    def test_a_windows_drive_letter_is_not_a_scheme(self):
        """`C:/data/x.tif` is a path, not scheme `C`.

        Test scenario:
            Splitting naively on the first colon makes every absolute Windows path unresolvable — and this
            package is developed on Windows, so it would fail immediately and confusingly.
        """
        original = resolvers()["file"]
        register_resolver("file", lambda uri: f"file-resolver:{uri}")
        try:
            assert resolve_uri("C:/data/x.tif") == "file-resolver:C:/data/x.tif", (
                "a drive letter must route to the file resolver"
            )
        finally:
            register_resolver("file", original)


class TestInMemoryObjects:
    """Referencing data a caller already holds, without writing it to disk."""

    def test_an_object_resolves_back_to_itself(self):
        """The reference reaches the same object, not a copy.

        Test scenario:
            The requirement that makes this usable from a notebook: a Dataset built in memory can back a
            figure without a round trip through the filesystem.
        """
        rows = [1, 2, 3]
        assert DataRef.to_object(rows, name="rows").open() is rows, (
            "an object reference must return the same object"
        )

    def test_reusing_an_id_replaces_what_it_points_at(self):
        """A stable id lets the data behind a figure be refreshed.

        Test scenario:
            Re-running a cell that rebuilds a Dataset should update the figure's source, not accumulate ids.
        """
        register_object([1], name="same-id")
        register_object([2], name="same-id")
        assert resolve_uri("object:same-id") == [2], "the later registration must win"

    def test_an_unregistered_id_says_why_it_cannot_resolve(self):
        """The error explains that the reference is process-local.

        Test scenario:
            A figure saved with an `object:` reference and reopened elsewhere fails here; "not found" alone
            would not tell the user that saving the data is the fix.
        """
        with pytest.raises(KeyError, match="only resolves in the process"):
            resolve_uri("object:never-registered")

    def test_clearing_releases_the_objects(self):
        """`clear_objects` drops the strong references the table holds.

        Test scenario:
            The table would otherwise keep every dataset a long session ever registered alive.
        """
        register_object([1], name="temp")
        clear_objects()
        with pytest.raises(KeyError):
            resolve_uri("object:temp")


class TestTheClassifierSeam:
    """`base/` declares how classification is reached; something above it supplies the arithmetic."""

    def test_a_classifier_is_registered_by_importing_the_package(self):
        """Importing `digitalearth` fills the seam, so `Scale` works with no setup.

        Test scenario:
            The seam exists because `base/` may not import cleopatra — not even lazily, since the
            engine-neutrality guard reads the source. If nothing filled it, every classified map would fail.
        """
        from digitalearth.base.registry import get_classifier

        assert callable(get_classifier()), (
            "importing the package must register a classifier"
        )

    def test_an_unfilled_seam_says_so_rather_than_reading_as_a_bad_scheme(self):
        """With nothing registered, `get_classifier` names the wiring, not the caller's arguments.

        Test scenario:
            The failure only happens when `digitalearth` was never imported or something replaced the
            registration — an install problem. A `KeyError` or an `AttributeError` here would send whoever
            hits it looking at their `scheme=`.
        """
        from digitalearth.base import registry

        saved = registry._CLASSIFIER.pop("fn")
        try:
            with pytest.raises(RuntimeError, match="no classifier is registered"):
                registry.get_classifier()
        finally:
            registry._CLASSIFIER["fn"] = saved


class TestTheFileResolver:
    """The built-in resolver: a path becomes a pyramids object, with the reader chosen by what is there."""

    #: Anchored on this file rather than the working directory, so the resolver is handed a real absolute
    #: path — which is the argument shape it exists to handle, and is what a stored figure would carry.
    _DATA = Path(__file__).resolve().parents[1] / "data"
    _EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "data"

    def test_a_raster_path_opens_as_a_dataset(self):
        """A GeoTIFF resolves through pyramids' raster reader.

        Test scenario:
            The common case, and the first reader tried. `DataRef("dem.tif").open()` is the whole point of
            the type — a layer naming its data instead of holding it.
        """
        from pyramids.dataset import Dataset

        opened = DataRef(str(self._EXAMPLES / "acc4000.tif")).open()
        assert isinstance(opened, Dataset), (
            f"a .tif must resolve to a pyramids Dataset, got {type(opened).__name__}"
        )

    def test_a_vector_path_opens_as_a_feature_collection(self):
        """A GeoJSON resolves through the vector reader, after the raster reader declines.

        Test scenario:
            The resolver does not switch on the extension — it tries the raster reader and falls through.
            That is deliberate (an extension lies often enough), and this is the path that proves the
            fall-through works rather than merely existing.
        """
        from pyramids.feature import FeatureCollection

        opened = DataRef(str(self._DATA / "points.geojson")).open()
        assert isinstance(opened, FeatureCollection), (
            f"a .geojson must resolve to a FeatureCollection, got {type(opened).__name__}"
        )

    def test_a_file_uri_resolves_the_same_as_a_bare_path(self):
        """The `file:` scheme is accepted as well as a plain path.

        Test scenario:
            A stored figure may carry either spelling, and a reference that resolved one way when written and
            another when read back would make the format useless.
        """
        path = self._DATA / "points.geojson"
        assert type(DataRef(f"file:{path}").open()) is type(
            DataRef(str(path)).open()
        ), "a file: URI and a bare path must reach the same reader"

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("file:///home/me/x.tif", "/home/me/x.tif"),
            ("file:///C:/data/x.tif", "C:/data/x.tif"),
            ("file:data/dem.tif", "data/dem.tif"),
            ("file:///a%20b/x.tif", "/a b/x.tif"),
        ],
    )
    def test_a_file_uri_keeps_the_path_it_names(self, uri, expected):
        """Every `file:` spelling resolves to the path it actually names.

        Args:
            uri: The reference as a stored figure would carry it.
            expected: The path it must resolve to, separators normalised.

        Test scenario:
            The three-slash form is RFC 8089's absolute spelling. Rescuing a Windows drive letter by stripping
            leading slashes turned every absolute POSIX path into a relative one, so `file:///home/me/x.tif`
            resolved against the working directory — opening the wrong file wherever a same-named relative
            path existed. CI's own matrix runs on Linux, and the old test used a Windows path with no `///`,
            so it never entered the branch.
        """
        from digitalearth.base.registry import _path_of

        assert _path_of(uri).replace("\\", "/") == expected, (
            f"{uri!r} must resolve to {expected!r}, got {_path_of(uri)!r}"
        )

    def test_a_missing_path_is_reported_as_missing(self):
        """A typo'd path blames the path, not the format.

        Test scenario:
            Both readers are tried in turn, so without this check a file that was never there surfaces as
            whatever the *vector* reader says about it — sending the user to check the format of a file that
            does not exist.
        """
        with pytest.raises(FileNotFoundError, match="no such file"):
            DataRef(str(self._DATA / "no-such-file.tif")).open()

    def test_something_readable_as_neither_blames_both_readers(self):
        """A file that exists but is neither raster nor vector reports both failures, chained.

        Test scenario:
            Reporting only the last error would blame the vector reader for a corrupt GeoTIFF. The chain is
            what lets whoever reads the traceback see that *both* readers were tried and why each declined.
        """
        with pytest.raises(Exception) as caught:
            DataRef(str(Path(__file__).resolve())).open()
        assert not isinstance(caught.value, FileNotFoundError), (
            "this file exists, so the failure must be a read failure, not a missing-path one"
        )
        assert caught.value.__cause__ is not None, (
            "the vector failure must chain from the raster failure, so both are visible"
        )


class TestRegisteringAResolver:
    """The guard on the registry itself."""

    def test_a_resolver_needs_a_scheme_to_be_reachable_under(self):
        """An empty scheme is refused rather than stored.

        Test scenario:
            `resolve_uri` derives the scheme from the URI and falls back to `"file"`, so nothing ever looks
            up `""`. A resolver registered there would be silently unreachable — which looks exactly like a
            plugin that failed to install.
        """
        with pytest.raises(ValueError, match="non-empty scheme"):
            register_resolver("", lambda uri: uri)
