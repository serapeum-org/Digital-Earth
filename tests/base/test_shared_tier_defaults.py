"""M4/M8/M9/L11 — the numbers and helpers the tiers are supposed to share are declared once, in ``base/``.

Four findings, one disease: a constant or a helper that every backend was meant to read got copied into each
backend instead, so the tiers could drift apart again — and in three of the four cases they already had.

* **M4** — ``DEFAULT_BASEMAP_PROVIDER`` reached the interactive and web tiers but not the *static* one, which
  kept handing cleopatra ``source=None`` and so drew ``OpenStreetMap.Mapnik``. The divergence the constant was
  added to kill survived on the default backend, the one installed without an extra.
* **M8** — ``50_000`` was a literal in ``interactive/base.py`` and ``web/base.py``, and only the web tier
  refused a negative override: the same bad call raised on one backend and routed every layer on the other.
* **M9** — ``DEFAULT_FPS = 3.0`` was declared in the static, 3-D and web tiers and written out twice more as a
  bare literal in the interactive one — five copies of one agreed number.
* **L11** — the colormap-sampling helper existed three times over (``interactive/vector.py``,
  ``three_d/base.py``, ``web/base.py``), so "one ``cmap`` colours identically everywhere" rested on three
  functions staying in step by hand.

Everything here runs in the lean ``dev`` environment: no test imports a rendering engine. The three tiers whose
engines are lazy are inspected by attribute, and the 3-D tier — whose modules import PyVista at module level —
is read with :mod:`ast` instead, which also catches a lazy import inside a function.
"""

import ast
import pathlib

import pytest

import digitalearth
from digitalearth.base.animation import DEFAULT_FPS
from digitalearth.base.basemaps import DEFAULT_BASEMAP_PROVIDER
from digitalearth.base.bigdata import (
    DEFAULT_BIG_DATA_THRESHOLD,
    validate_big_data_threshold,
)
from digitalearth.base.symbology import sample_cmap

SRC = pathlib.Path(digitalearth.__file__).parent

#: The Carto tile set every tier requests for the shared default provider. Comparing whole URLs would fail on
#: cosmetic differences each engine introduces (a subdomain shard, upper-cased ``{Z}/{X}/{Y}`` placeholders);
#: the ``light_all`` path segment is the tile set itself, which is what "the same basemap" means.
CARTO_LIGHT = "light_all"


def _tree(relative: str) -> ast.Module:
    """Parse a package module without importing it.

    Args:
        relative: Path of the module under ``src/digitalearth``, ``/``-separated.

    Returns:
        The parsed module.
    """
    return ast.parse((SRC / relative).read_text(encoding="utf-8"))


def _assigned_names(tree: ast.Module) -> set[str]:
    """Return the module-level names ``tree`` assigns, annotated assignments included.

    Args:
        tree: A parsed module.

    Returns:
        Every name bound by a top-level ``name = ...`` or ``name: T = ...``.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return names


def _imports_from(tree: ast.Module, module: str) -> set[str]:
    """Return the names ``tree`` imports from ``module``, lazy imports inside functions included.

    Args:
        tree: A parsed module.
        module: The dotted module path to look for.

    Returns:
        Every name imported from ``module``.
    """
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def _function_names(tree: ast.Module) -> set[str]:
    """Return every function/method name defined anywhere in ``tree``.

    Args:
        tree: A parsed module.

    Returns:
        The defined function names.
    """
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _resolve_threshold(tier: str, threshold: int) -> int:
    """Run one tier's per-call ``big_data_threshold`` through its own resolver.

    Both tiers construct without their engine (the maps' imports are lazy), so this reaches the real
    resolvers rather than a stub — which is what makes "both tiers answer the same way" a claim about the
    shipped code.

    Args:
        tier: ``"interactive"`` or ``"web"``.
        threshold: The per-call override to resolve.

    Returns:
        The cutoff the tier resolved it to.
    """
    if tier == "interactive":
        from digitalearth.interactive.map import InteractiveMap

        return InteractiveMap()._resolve_big_data_threshold(
            threshold, None, caller="InteractiveMap.points()"
        )
    from digitalearth.web.map import WebMap

    return WebMap()._threshold(threshold, caller="WebMap.points()")


class TestM4TheBasemapDefaultReachesEveryTier:
    """The shared provider constant is read by all three tile-drawing tiers, static included."""

    def test_the_static_tier_resolves_the_shared_default(self):
        """``basemap()`` with no argument means the shared provider, not cleopatra's own.

        Test scenario:
            The static tier used to pass ``source=None`` straight through, so cleopatra picked its own
            default. Resolving ``None`` here is what makes the omitted argument mean the same thing on this
            tier as the keyword default does on the other two.
        """
        from digitalearth.static.maps.decoration import (
            _SHARED_PROVIDERS,
            _resolve_tile_source,
        )

        assert (
            _resolve_tile_source(None)
            == _SHARED_PROVIDERS[DEFAULT_BASEMAP_PROVIDER.lower()]
        ), "the static tier no longer resolves the shared default"

    def test_the_static_tier_no_longer_falls_through_to_cleopatra(self):
        """The resolved provider is a different tile service from the one cleopatra would have chosen.

        Test scenario:
            This is the finding itself: ``quickmap(basemap=True)`` drew OpenStreetMap on the default backend
            and Carto on the other two. Comparing the resolved provider against cleopatra's own default is
            what fails if the resolution is ever removed again.
        """
        from cleopatra.basemap.tiles import get_provider

        from digitalearth.static.maps.decoration import _resolve_tile_source

        cleopatras_own = get_provider(None)
        assert cleopatras_own.name == "OpenStreetMap.Mapnik", (
            "cleopatra's default changed; this test's premise needs rechecking"
        )
        assert get_provider(_resolve_tile_source(None)).name != cleopatras_own.name

    def test_the_static_and_web_tiers_request_the_same_tiles(self):
        """Both tiers' default basemap requests name the same Carto tile set.

        Test scenario:
            Each tier spells the provider for its own engine — an ``xyzservices`` path here, a URL template
            on the web tier — so the agreement can only be checked on what is actually fetched. The
            interactive tier's half of this needs GeoViews and lives in ``tests/interactive``.
        """
        from cleopatra.basemap.tiles import get_provider

        from digitalearth.static.maps.decoration import _resolve_tile_source
        from digitalearth.web.decoration import _BASEMAP_PROVIDERS

        static_url = get_provider(_resolve_tile_source(None)).build_url()
        web_url = _BASEMAP_PROVIDERS[DEFAULT_BASEMAP_PROVIDER.lower()][0]
        assert CARTO_LIGHT in static_url, static_url
        assert CARTO_LIGHT in web_url, web_url

    def test_a_shared_name_is_translated_not_rejected(self):
        """The four cross-tier names resolve on the static tier too, case-insensitively.

        Test scenario:
            The other tiers take ``"CartoDark"``/``"OSM"`` by name. Passing one here used to reach
            ``xyzservices`` as an unknown provider, so the same argument worked on two tiers and raised on the
            third — the same divergence as the default, one step along.
        """
        from digitalearth.static.maps.decoration import _resolve_tile_source

        assert _resolve_tile_source("cartodark") == "CartoDB.DarkMatter"
        assert _resolve_tile_source("OSM") == "OpenStreetMap.Mapnik"

    def test_an_unrelated_provider_passes_through_untouched(self):
        """Anything that is not a shared name reaches cleopatra exactly as written.

        Test scenario:
            The translation must be additive: an ``xyzservices`` path, a ``TileProvider`` or a URL was always
            legal here and still is.
        """
        from digitalearth.static.maps.decoration import _resolve_tile_source

        url = "https://a.tile.example/{z}/{x}/{y}.png"
        assert _resolve_tile_source("Esri.WorldImagery") == "Esri.WorldImagery"
        assert _resolve_tile_source(url) == url


class TestM8TheBigDataCutoffIsOneNumberAndOneRule:
    """The threshold is declared once, and both tiers refuse the same bad override."""

    def test_both_tiers_read_the_shared_constant(self):
        """Neither tier keeps a literal ``50_000`` of its own.

        Test scenario:
            Two literals meant either tier could be retuned alone, silently moving the size at which a layer
            changes rendering strategy on one backend only.
        """
        from digitalearth.interactive.base import (
            DEFAULT_BIG_DATA_THRESHOLD as interactive_default,
        )
        from digitalearth.web.base import DEFAULT_BIG_DATA_THRESHOLD as web_default

        assert interactive_default is DEFAULT_BIG_DATA_THRESHOLD
        assert web_default == DEFAULT_BIG_DATA_THRESHOLD

    def test_the_interactive_tier_imports_it_rather_than_declaring_it(self):
        """The constant is imported from ``base/``, not re-assigned in the tier.

        Test scenario:
            Reading the value alone cannot tell a shared constant from a literal that happens to match, which
            is exactly the state the finding described. Parsing the module is what distinguishes them.
        """
        tree = _tree("interactive/base.py")
        assert "DEFAULT_BIG_DATA_THRESHOLD" not in _assigned_names(tree)
        assert "DEFAULT_BIG_DATA_THRESHOLD" in _imports_from(
            tree, "digitalearth.base.bigdata"
        )

    @pytest.mark.parametrize("tier", ["interactive", "web"])
    def test_both_tiers_refuse_a_negative_cutoff(self, tier):
        """A negative per-call override raises the same error on either backend.

        Args:
            tier: Which tier's resolver to exercise.

        Test scenario:
            Web raised and interactive silently accepted it, so ``big_data_threshold=-1`` — a plausible
            "unlimited" sentinel borrowed from another API — routed *every* layer, empty ones included,
            through Datashader without a word.
        """
        with pytest.raises(ValueError, match="must not be negative"):
            _resolve_threshold(tier, -1)

    @pytest.mark.parametrize("tier", ["interactive", "web"])
    def test_both_tiers_accept_a_zero_cutoff(self, tier):
        """Zero is legal on both — it routes every non-empty layer, which is a real request.

        Args:
            tier: Which tier's resolver to exercise.

        Test scenario:
            The guard must reject only the nonsensical value. A shared rule that quietly tightened what one
            tier accepted would be the same drift in the other direction.
        """
        assert _resolve_threshold(tier, 0) == 0

    def test_the_error_names_the_call_that_made_it(self):
        """The message points at the builder, not at the shared helper.

        Test scenario:
            A shared guard is only usable if its error still reads like the tier's own; otherwise each tier
            re-implements the check to get a decent message, which is how the duplication started.
        """
        with pytest.raises(ValueError, match=r"WebMap\.points\(\)"):
            validate_big_data_threshold(-5, caller="WebMap.points()")


class TestM9TheAnimationRateIsOneConstant:
    """Every tier's animation entry point starts from the same frames-per-second value."""

    @pytest.mark.parametrize(
        ("module", "attribute"),
        [
            ("digitalearth.static.maps.animation", "DEFAULT_FPS"),
            ("digitalearth.interactive.animation", "DEFAULT_FPS"),
            ("digitalearth.web.export", "DEFAULT_FPS"),
        ],
        ids=["static", "interactive", "web"],
    )
    def test_each_tier_exposes_the_shared_constant(self, module, attribute):
        """The name each tier's callers reach for is the shared object itself.

        Args:
            module: The tier module to import.
            attribute: The constant name on it.

        Test scenario:
            ``is`` rather than ``==`` — a re-declared literal of the same value would pass an equality check
            while being free to drift on the next edit, which is the finding.
        """
        import importlib

        assert getattr(importlib.import_module(module), attribute) is DEFAULT_FPS

    @pytest.mark.parametrize(
        ("method_path", "name"),
        [
            ("digitalearth.static.map:Map.animate", "fps"),
            ("digitalearth.static.map:Map.rotate", "fps"),
            ("digitalearth.interactive.map:InteractiveMap.play", "fps"),
            ("digitalearth.interactive.map:InteractiveMap.save_animation", "fps"),
        ],
        ids=["static-animate", "static-rotate", "interactive-play", "interactive-save"],
    )
    def test_every_entry_point_defaults_to_it(self, method_path, name):
        """Each animation entry point's signature default *is* the shared rate.

        Args:
            method_path: ``module:Class.method`` to inspect.
            name: The parameter carrying the rate.

        Test scenario:
            The interactive tier wrote ``fps: float = 3.0`` inline on both of its entry points, so the number
            agreed with the others only by coincidence; nothing linked them.
        """
        import importlib
        import inspect

        module_name, qualname = method_path.split(":")
        target = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
        assert inspect.signature(target).parameters[name].default is DEFAULT_FPS

    def test_the_three_d_tier_imports_it_rather_than_declaring_it(self):
        """The 3-D tier reads the shared constant too, checked without importing PyVista.

        Test scenario:
            ``three_d/animation.py`` held its own ``DEFAULT_FPS: float = 3.0``, the third declaration of one
            agreed number. Its module cannot be imported in this environment, so the source is parsed instead.
        """
        tree = _tree("three_d/animation.py")
        assert "DEFAULT_FPS" not in _assigned_names(tree)
        assert "DEFAULT_FPS" in _imports_from(tree, "digitalearth.base.animation")

    @pytest.mark.parametrize(
        "module",
        [
            "static/maps/animation.py",
            "interactive/animation.py",
            "three_d/animation.py",
            "web/export.py",
        ],
    )
    def test_no_tier_declares_a_rate_of_its_own(self, module):
        """No animation module assigns ``DEFAULT_FPS`` at all — every one imports it.

        Args:
            module: The tier module to parse.

        Test scenario:
            This is the guard that actually holds the finding closed: a future edit that re-adds a local
            default to any tier fails here rather than in a viewer's playback speed.
        """
        assert "DEFAULT_FPS" not in _assigned_names(_tree(module))

    def test_the_save_time_fallback_is_a_separate_name(self):
        """The encoder's unknown-rate fallback no longer shares the entry points' name.

        Test scenario:
            ``static/animation.py`` declares a *different* concept — the rate to encode at when the clip's own
            rate is unknown — and it was also called ``DEFAULT_FPS``, at a different value. Two names now, so
            neither reads as a copy of the other.
        """
        from digitalearth.static import animation

        assert not hasattr(animation, "DEFAULT_FPS"), (
            "the save-time fallback must not share the entry points' constant name"
        )
        assert animation.FALLBACK_SAVE_FPS == 12.0
        assert animation.FALLBACK_SAVE_FPS != DEFAULT_FPS


class TestL11TheColormapSamplerIsShared:
    """One colormap-sampling helper, in ``base/symbology.py``, read by the tiers that need literal colours."""

    def test_the_interactive_tier_uses_the_shared_sampler(self):
        """``interactive/vector.py`` imports it and defines no copy.

        Test scenario:
            Its ``_cmap_hex`` was a stop-for-stop copy of the web tier's, with a docstring promising they
            matched — a promise nothing enforced.
        """
        tree = _tree("interactive/vector.py")
        assert "sample_cmap" in _imports_from(tree, "digitalearth.base.symbology")
        assert "_cmap_hex" not in _function_names(tree)

    def test_the_three_d_tier_uses_the_shared_sampler(self):
        """``three_d/base.py`` imports it and defines no copy, checked without importing PyVista.

        Test scenario:
            Its copy was spelled ``_sample_cmap`` and accepted an already-built colour sequence as well as a
            name — the superset the shared helper now offers every tier.
        """
        tree = _tree("three_d/base.py")
        assert "sample_cmap" in _imports_from(tree, "digitalearth.base.symbology")
        assert "_sample_cmap" not in _function_names(tree)

    def test_the_remaining_web_copy_still_agrees_stop_for_stop(self):
        """``WebMapBase._cmap_hex`` samples exactly as the shared helper does.

        Test scenario:
            The web tier's copy is the one collapse still outstanding (``web/base.py`` was being edited
            elsewhere when the other two were rewired). Until it is folded in, this pins the agreement the
            three copies were supposed to have: a mismatch means a ``column``/``scheme``/``k`` triple colours
            differently on the web tier than on the other two.
        """
        from digitalearth.web.base import WebMapBase

        for n in (1, 2, 5):
            assert WebMapBase._cmap_hex("viridis", n) == sample_cmap("viridis", n)

    def test_a_single_class_takes_the_middle_of_the_ramp(self):
        """One class is coloured from the ramp's midpoint, not its dark end.

        Test scenario:
            All three copies carried this special case; sharing it is what stops one of them losing it, which
            would paint a single-class layer in a near-black that reads as missing data.
        """
        assert sample_cmap("viridis", 1) == [sample_cmap("viridis", 3)[1]]
