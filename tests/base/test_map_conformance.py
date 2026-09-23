"""One set of behavioural questions every tier's map answers the same way (U-4, #323).

The conformance suite had a name half and no behavioural half. `tests/test_contract_names.py`,
`tests/base/test_contract.py` and `tests/base/test_capabilities.py` hold every facade to CORE / ALIASES /
PENDING and to its own `Capabilities`; `tests/base/test_renderer_conformance.py` holds all four renderers to
one behaviour. Between them nothing asked the question a caller actually asks: **does the same call, on any
tier, produce the same described result?** A tier can answer to every Core name, with every declared keyword,
and still describe a different figure than its neighbour — because a name test reads a signature and never
makes the call.

So this is the other half. A tier supplies three things — its name, a way to hand back an empty map, and
whatever it legitimately cannot do — and inherits every probe below. **Adding a tier is a subclass**, and
the two that sign here are a `backend` string and a `make()` and nothing else. The probes never mention a
tier: they call the Core names (`points`, `choropleth`, `graticule`) and read `figure_spec`, which is the
whole point of the contract those names were frozen into.

**Which tiers subclass it, and why the other two cannot yet.** The package has four —
:data:`ALL_TIERS` — and `web` and `interactive` are the two that sign here. The **static** tier is the
*default* one and is still absent: `points` and `choropleth` are `PENDING` under the Core spelling on it
(they are `scatter`, and a `choropleth` with a different signature), so probes written against the frozen
names cannot run against it until order 27a. The **3-D** tier draws a scene rather than a map and has no
`points`/`graticule` at all. Saying that nowhere is what let :data:`NO_PORTABLE_CHANNELS` sit empty beside
two tiers that record no portable channel, reading as a statement about the package when it was one about
the subset (review R-M4). Both are named in the tables now, and
:class:`TestTheTablesDescribeThePackage` re-asks them the questions those tables answer for — so an entry
cannot outlive the defect it excuses on an unsubclassed tier either (review R-L7).

**About `to_backend()`.** The brief this was written from calls for a minimal `to_backend` round trip, web to
interactive. `to_backend()` does not exist yet — it is **order 33 (U-6)**, and building it is explicitly not
this module's job. What is built here is the *probe that will hold it*, exercised today against the two
tiers' `figure_spec`s directly. A round trip through `to_backend()` is only worth having if the two ends
describe the same figure, so these probes are the precondition: each tier builds the shared seed figure
(:func:`_seed`) and is held to one described result, :data:`EXPECTED_SEED`. Held to the same constant, the
tiers are held to each other — which is what lets each half run in its own CI job, where only one of them
collects: `web` carries MapLibre, `interactive` carries HoloViz, and neither of those two carries the other.
The `all` environment does carry both, and :class:`TestOneStyledLayerDescribesOneChannelOnBothTiers` uses
it to ask the two live tiers directly; the constant is what holds them in the jobs that cannot. When
`to_backend()` lands it gets one more probe — that a figure carried across describes what
:data:`EXPECTED_SEED` already pins — and nothing here changes.

**What a described result is, and what it deliberately leaves out.** :func:`_described` reduces a figure to
`(kind, band, visible)` per layer in draw order. It does **not** compare `symbology.props`, because there is
nothing there to compare: the web tier records MapLibre's own paint
(`{"maplibre_type": "circle", "paint": {"circle-radius": 7.0, ...}}`) and the interactive tier records
HoloViews' (`{"via": "geometry", "hv_type": "Points", "common": {"size": 7.0}, ...}`). The vocabulary's
portable answer to the same question — `Symbology.encodings`, keyed by
:data:`~digitalearth.base.spec.encoding.CHANNELS` — was written by **neither tier**; both came back with
`encodings == {}`, so a figure could round-trip its kinds, its draw order and its visibility and then be
redrawn in the target engine's default colours and sizes. Both 2-D tiers record the declared channels now
(#328) and are off :data:`NO_PORTABLE_CHANNELS`, which names the static and 3-D tiers instead, and
:class:`TestOneStyledLayerDescribesOneChannelOnBothTiers` asks the two live tiers for the same styled layer
and holds them to one set of channels.

The seed figure still carries no style, and that is deliberate: it draws `points()` with no arguments, and
the tiers' *defaults* differ — the web tier's marker is 5 pixels and the interactive tier's is 6. Which
default a tier picks is not what a round trip has to agree about; what a caller **asked** for is, so the
styled probe passes an explicit value rather than comparing two silences — and
:meth:`MapConformanceBase.test_an_unstyled_layer_publishes_no_portable_channel` holds the tiers to the same
rule in `encodings`, which briefly published each tier's defaults as though a caller had asked for them
(review R-H2).

The classification is the other thing the two tiers are held to, and it is **not** a symbology value:
`choropleth(..., scheme=, k=)` goes through :class:`~digitalearth.base.spec.scale.Scale` on every tier and
both publish the result as `last_breaks`, an attribute of the live map that no `FigureSpec` carries.
:data:`EXPECTED_BREAKS` pins it as live-map parity between two engines — not as part of the `to_backend()`
precondition, which it was framed as while reading something a figure does not carry (review R-M6). What
the figure does carry is asked separately and guarded by :data:`NO_PORTABLE_CLASSIFICATION`.

**The collection trap this module is built not to fall into.** The shared renderer contract collected nothing
useful in the 3-D and interactive jobs for its whole life: its classes are gated on optional engines, and the
only job collecting `tests/base` had neither installed, so every check it declared was skipped. `test-3d` and
`test-interactive` were fixed by naming that module in the task's own command. The same fix is applied here —
and, because a fix nobody checks is a fix with a shelf life, :class:`TestEveryTierIsReallyCollected` asserts
it three ways: the tier's task names this module, the tier's environment carries the engine that ungates it,
and a real `--collect-only` run reports a non-zero item count for every tier whose engine is importable. The
first two run in **every** environment, the plain `dev` matrix included, so they cannot themselves go
uncollected.
"""

import importlib.util
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from digitalearth.base.registry import band_of

#: This module, spelled the way a pixi task and a pytest node id spell it. Both the task guard and the
#: collection guard read it, so the path is written once.
MODULE_PATH = "tests/base/test_map_conformance.py"

#: The repository root. `tests/base/<this file>` — two parents up. Resolved from `__file__` rather than from
#: the working directory, so the guards below answer the same whichever directory pytest was started in.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The kinds the seed figure draws, by the engine-neutral name every tier files them under.
POINTS = "points"
CHOROPLETH = "choropleth"
GRATICULE = "graticule"

#: The bands those kinds are drawn in. A graticule is a reference, drawn beneath the data whenever it was
#: added; points and a choropleth are data.
DATA_BAND = "data"
REFERENCE_BAND = "reference"

#: The column, scheme and class count the classification probe asks for. One spelling, used by both tiers,
#: because the promise being checked is that one call classifies one way.
COLUMN = "pop"
SCHEME = "quantiles"
CLASSES = 2

#: The marker size the style probe asks for. `size` is a declared keyword of `points` in the Core contract
#: and a declared channel in :data:`~digitalearth.base.spec.encoding.CHANNELS`, so a tier has both a reason
#: to accept it and a place to record it — which is what made its absence from `Symbology.encodings` worth
#: asserting rather than assuming.
SIZE = 7.0

#: The opacity and the colour the cross-tier probe asks for alongside `SIZE`, so three channels of three
#: different kinds are held rather than one number: a number the tiers spell differently, and a colour.
OPACITY = 0.25
COLOUR = "#cc4444"

#: The same three channels, in each tier's own keyword spelling.
#:
#: The web tier's opacity keyword is `opacity=` and the interactive tier's is `alpha=`. That disagreement is
#: filed as #332 and is **not** reconciled here: the promise this probe holds is that the two describe one
#: *channel*, whatever each calls the keyword that sets it, so the two spellings sit side by side.
CROSS_TIER_STYLE: dict[str, dict] = {
    "web": {"size": SIZE, "opacity": OPACITY, "color": COLOUR},
    "interactive": {"size": SIZE, "alpha": OPACITY, "color": COLOUR},
}

#: What either tier must say the styled layer draws: channel -> the value the caller asked for, sorted.
EXPECTED_CHANNELS = (("color", COLOUR), ("opacity", OPACITY), ("size", SIZE))

#: The cross-backend job, as `(pixi task, pixi environment)`. The only job with both 2-D engines in one
#: process, and so the only one where :class:`TestOneStyledLayerDescribesOneChannelOnBothTiers` runs live.
CROSS_TIER_JOB = ("test-backends", "all")

#: The class edges `COLUMN`/`SCHEME`/`CLASSES` cut :func:`_polygons` into. Two polygons valued 1 and 9, in
#: two quantile classes: the minimum, the break, the maximum. Both tiers route the request through
#: :class:`~digitalearth.base.spec.scale.Scale`, so both must publish these — which is the one symbology
#: value portable enough to compare across tiers today.
EXPECTED_BREAKS = (1.0, 5.0, 9.0)

#: What :func:`_seed` must describe, on any tier: `(kind, band, visible)` per layer, in draw order.
#:
#: The seed adds points, then a choropleth, then a graticule. The graticule comes **first** here because its
#: kind is drawn in the reference band, under the data, whenever it was asked for; the points and the
#: choropleth keep the order they were added in, because they share a band. So this one constant pins all
#: three questions the brief asks of a round trip — the kinds drawn, the draw order, and the visibility.
EXPECTED_SEED = (
    (GRATICULE, REFERENCE_BAND, True),
    (POINTS, DATA_BAND, True),
    (CHOROPLETH, DATA_BAND, True),
)

#: The tiers that cannot build a layer already hidden, each with what they do instead.
#:
#: A drift guard, not an excuse: :meth:`MapConformanceBase.test_a_tier_on_the_hidden_at_build_list_still_belongs_there`
#: builds the hidden layer on every listed tier and **fails when the tier has been fixed**, so an entry cannot
#: outlive what it excuses — the same shape as `UNDRAWN_KINDS` in the renderer contract. Taking a tier off the
#: list is what turns its `test_a_layer_built_hidden_is_described_hidden` on.
#:
#: **The interactive tier came off it, and the 3-D tier went on.** The interactive tier was listed here:
#: `points(visible=False)` fell through the builder's `**opts` to HoloViews as a style option, so the element
#: was hidden while the figure went on describing the layer visible (#327). The flag is a declared parameter
#: of every builder on that tier now, so it reaches `add_element(visible=)` and the description and the
#: drawing say the same thing. The 3-D tier never gained it: measured,
#: `Scene3D().point_cloud(cloud, visible=False)` raises `TypeError: "visible" is an invalid keyword argument
#: for _common_arg_parser` — the flag falls through `**kwargs` to PyVista, which refuses it, and
#: `set_visible()` after the build is the only way to hide a 3-D layer. The static tier is **not** listed:
#: measured, `Map(crs=4326).scatter(features, visible=False)` describes the layer `visible=False` (R-M4).
#:
#: Listing the 3-D tier is also what makes the reverse branch execute again. With the table empty the guard
#: skipped for every tier, in every job, so nothing was being guarded in either direction (review R-L7); the
#: tiers named here that do not subclass the base are held by
#: :class:`TestTheTablesDescribeThePackage` instead.
CANNOT_HIDE_AT_BUILD: dict[str, str] = {
    "3d": (
        "visible= is not a parameter of its builders; it reaches PyVista through **kwargs, which raises "
        "TypeError, so a 3-D layer is hidden with set_visible() after it is built"
    ),
}

#: The tiers that record no portable channel for a layer's style, each with what they record instead.
#:
#: `Symbology.encodings` is the vocabulary's answer to "what drives this layer's colour, size and opacity",
#: keyed by :data:`~digitalearth.base.spec.encoding.CHANNELS` so any tier can read any other tier's.
#:
#: **The 2-D tiers came off it; the other two never did.** Both 2-D tiers were listed: each recorded its
#: engine's own spelling — MapLibre's paint under `props['paint']`, HoloViews' resolved options under
#: `props['common']` and `props['opts']` — and neither wrote an encoding, so a styled layer described its
#: style in a form only the tier that drew it could read. Both lift the declared channels out of what they
#: already record now (#328), and because the list is guarded in **both** directions a fixed tier left on it
#: fails just as a broken tier taken off it would.
#:
#: The static and 3-D tiers were never fixed, and the empty table said otherwise (review R-M4). Measured:
#: `Map(crs=4326).scatter(features, size=7.0)` records `props == ['opts', 'size_column', 'via']` and
#: `encodings == {}`; `Scene3D().point_cloud(cloud, size=7.0)` records `size` flat in `props` and
#: `encodings == {}`. Neither is a 2-D map, and neither has a fold that turns its own spelling into declared
#: channels, so both are named here with what they record instead — and both are held to it, in both
#: directions, by :class:`TestTheTablesDescribeThePackage`.
NO_PORTABLE_CHANNELS: dict[str, str] = {
    "matplotlib": (
        "the caller's keywords stay in props['opts'] and go to cleopatra as they are; nothing on this tier "
        "folds them into declared channels"
    ),
    "3d": (
        "the builders record their resolved keywords flat in props — size, cmap, scheme — with no fold onto "
        "declared channels"
    ),
}

#: Each tier's CI job, as `(pixi task, pixi environment, the pixi feature carrying its engine)`.
#:
#: Read by :class:`TestEveryTierIsReallyCollected`, which is the whole reason this is written down: the
#: renderer contract's classes were gated on an engine no collecting job had, and nothing said so. A tier is
#: really covered only when its task names this module **and** the environment that task is run in carries
#: the feature that installs its engine.
TIER_JOBS: dict[str, tuple[str, str, str]] = {
    "web": ("test-web", "web", "web"),
    "interactive": ("test-interactive", "interactive", "interactive"),
}

#: Each tier's engine, as the module whose absence gates the tier's class. The collection guard asks whether
#: it is importable to decide whether this environment is one where that tier must contribute items.
TIER_ENGINES: dict[str, str] = {"web": "maplibre", "interactive": "geoviews"}

#: The tiers whose **figure** carries no class edges, each with what carries them instead.
#:
#: `choropleth(..., scheme=, k=)` classifies through :class:`~digitalearth.base.spec.scale.Scale` on every
#: tier, and both tiers publish the result on the live map as `last_breaks`. A `FigureSpec` carries neither:
#: measured, the breaks appear in no layer's `symbology.props` and in no `encodings`, on either tier, and the
#: probe that read `last_breaks` was framed as the `to_backend()` precondition while reading something a
#: figure does not carry (review R-M6).
#:
#: Guarded in both directions by
#: :meth:`MapConformanceBase.test_a_tier_whose_figure_carries_no_class_edges_is_named_as_such`, so the entry
#: retires itself the day a tier publishes the `Scale` on its colour encoding.
#:
#: This table is about the two tiers that record encodings at all. The static and 3-D tiers are excused
#: wholesale by :data:`NO_PORTABLE_CHANNELS` — a tier that publishes no encoding publishes no scale either.
NO_PORTABLE_CLASSIFICATION: dict[str, str] = {
    "web": (
        "the edges become a MapLibre step expression in paint['fill-color'], which carries the interior "
        "stop (5.0) and neither end; the expression is refused as a colour, so the layer publishes no "
        "colour encoding to hang a Scale on"
    ),
    "interactive": (
        "the edges become props['common']['color_levels'] beside a color naming the value dimension, and a "
        "colour that names a column is refused, so the layer publishes no colour encoding either"
    ),
}

#: Every tier the package ships, as its own `Capabilities` spells the backend.
#:
#: Written out because the tables above describe a *subset* and nothing said so. Two of these four subclass
#: :class:`MapConformanceBase`; the other two are held to the drift tables instead, and
#: :class:`TestTheTablesDescribeThePackage` refuses a tier that is in neither place (review R-M4).
ALL_TIERS: tuple[str, ...] = ("3d", "interactive", "matplotlib", "web")

#: The name the naming probes ask for, and what a second layer asking for it again must be given.
#:
#: One constant per spelling because both probes read both: the first checks that the name a caller gives is
#: the id they get back, the second that a name used twice is *suffixed* rather than shared — the answer all
#: four tiers now give, through :func:`~digitalearth.base.spec.layer.free_layer_id` (#321).
ASKED_NAME = "wells"
SUFFIXED_NAME = "wells-2"

# ---------------------------------------------------------------------------------------------------------
# The seam D-11 left behind, and closed. `points()` took no `name=` on either 2-D tier — the web tier's did,
# the interactive tier's swallowed it into `**opts` — so the probe that the id a caller asked for is the id
# the figure describes could not be written, and was reserved for here (#321). It is written now, directly
# beneath `test_two_layers_are_described_in_the_order_they_were_added`, reading `figure.layers.ids` against
# the names passed in.
#
# What has *not* changed is `_described`, which still compares the seed by kind and carries no ids. Those are
# two different promises: that a caller's own name is honoured (the probes below, on a named layer), and that
# an *unnamed* layer is numbered the same way everywhere — which it is not, since the web tier numbers a
# points layer after the MapLibre type it draws (`circle-1`) and the interactive tier after the kind
# (`points-1`). Neither spelling is wrong, and settling which one every tier generates is not what naming a
# layer was about.
# ---------------------------------------------------------------------------------------------------------


def _points():
    """Return the point features every tier's `points()` probe draws.

    A pyramids `FeatureCollection` in EPSG:4326 rather than a bare GeoDataFrame, so each builder really goes
    through its reproject-to-display-CRS path — the interactive tier's display CRS is Web Mercator and the
    web tier's is EPSG:4326, and a probe that skipped that path would not be drawing what a caller draws.

    Returns:
        A four-point `FeatureCollection` with one numeric column.
    """
    import geopandas as gpd
    from pyramids.feature import FeatureCollection
    from shapely.geometry import Point

    return FeatureCollection(
        gpd.GeoDataFrame(
            {"value": [1.0, 2.0, 3.0, 4.0]},
            geometry=[
                Point(4.9, 52.4),
                Point(5.1, 52.1),
                Point(4.5, 52.0),
                Point(5.5, 52.9),
            ],
            crs=4326,
        )
    )


def _polygons():
    """Return the polygon features the classification probe cuts into classes.

    Two squares valued 1 and 9. Small on purpose: the point of the probe is the class edges, and two values
    in two quantile classes give edges a reader can check by hand against :data:`EXPECTED_BREAKS`.

    Returns:
        A two-polygon `FeatureCollection` with the `COLUMN` attribute.
    """
    import geopandas as gpd
    from pyramids.feature import FeatureCollection
    from shapely.geometry import Polygon

    return FeatureCollection(
        gpd.GeoDataFrame(
            {COLUMN: [1.0, 9.0]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
            crs=4326,
        )
    )


def _point_cloud():
    """Return the three points the 3-D tier's probe draws.

    Returns:
        An `(n, 3)` array of x/y/z coordinates. Three points because the probe is about what the tier
        *records*, not about what it renders, and three is the smallest cloud PyVista will take.
    """
    import numpy as np

    return np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 0.5, 3.0]])


def _static_map():
    """Return an empty static map in the display CRS the other tiers' probes use.

    Returns:
        The matplotlib tier's `Map`.
    """
    from digitalearth.static import Map

    return Map(crs=4326)


def _static_styled(drawn) -> None:
    """Draw one explicitly styled point layer on the static tier.

    Args:
        drawn: The static map to draw on.

    Note:
        `scatter`, not `points`: the Core spelling is `PENDING` on this tier, which is the reason it does
        not subclass :class:`MapConformanceBase` and is held here instead.
    """
    drawn.scatter(_points(), size=SIZE)


def _static_hidden(drawn) -> None:
    """Ask the static tier for a point layer that starts hidden.

    Args:
        drawn: The static map to draw on.
    """
    drawn.scatter(_points(), visible=False)


def _scene_3d():
    """Return an empty 3-D scene.

    Returns:
        The 3-D tier's `Scene3D`.
    """
    from digitalearth.three_d import Scene3D

    return Scene3D()


def _cloud_styled(drawn) -> None:
    """Draw one explicitly styled point cloud on the 3-D tier.

    Args:
        drawn: The scene to draw on.
    """
    drawn.point_cloud(_point_cloud(), size=SIZE)


def _cloud_hidden(drawn) -> None:
    """Ask the 3-D tier for a point cloud that starts hidden.

    Args:
        drawn: The scene to draw on.
    """
    drawn.point_cloud(_point_cloud(), visible=False)


@dataclass(frozen=True)
class OutsideTheSuite:
    """How to reach a tier that does not subclass :class:`MapConformanceBase`.

    The tables above excuse a tier from a promise, and an excuse is only honest while something re-asks the
    question. The subclassed tiers are re-asked by the base class; these two had nothing asking, so an entry
    for them would have been a claim rather than a measurement (review R-M4/R-L7).

    Attributes:
        engine: The module whose absence means this tier cannot be exercised here, for `importorskip`.
        make: Returns an empty map or scene of that tier.
        styled: Draws one layer on it with an explicit `size=`, so there is a channel to record.
        hidden: Asks it for a layer that starts hidden.
    """

    engine: str
    make: Callable[[], Any]
    styled: Callable[[Any], None]
    hidden: Callable[[Any], None]


#: The two tiers the probes above do not reach, and how to ask them the questions the tables answer for.
#:
#: The static tier's `points`/`choropleth` are `PENDING` under the Core spelling (they are `scatter` and
#: `choropleth` with a different signature), and the 3-D tier draws a scene rather than a map — neither can
#: inherit probes written against the frozen Core names. That is why they are not subclasses, and it is
#: why they need this.
ABSENT_TIERS: dict[str, OutsideTheSuite] = {
    "matplotlib": OutsideTheSuite(
        "matplotlib", _static_map, _static_styled, _static_hidden
    ),
    "3d": OutsideTheSuite("pyvista", _scene_3d, _cloud_styled, _cloud_hidden),
}


def _closed(drawn) -> None:
    """Release a map built outside the `drawn` fixture.

    Args:
        drawn: The map or scene to close.
    """
    try:
        drawn.close()
    except Exception:  # pragma: no cover - a closed map may refuse a second close
        pass


def _asked_to_hide(tier: OutsideTheSuite, drawn) -> str:
    """Ask one tier for a layer that starts hidden, and say what it did about it.

    Args:
        tier: The tier's row in :data:`ABSENT_TIERS`.
        drawn: Its empty map or scene.

    Returns:
        `"refused"` when the builder raised rather than take the flag, `"hidden"` when the figure describes
        the layer hidden, and `"visible"` when it accepted the flag and described the layer visible anyway.
        Three answers rather than two because a tier on :data:`CANNOT_HIDE_AT_BUILD` can fail either way,
        and only `"hidden"` means the entry has to come off.
    """
    try:
        tier.hidden(drawn)
    except TypeError:
        return "refused"
    figure = drawn.figure_spec
    shown = figure.layers.get(figure.layers.ids[-1]).visible
    return "visible" if shown else "hidden"


def _class_edges(symbology) -> tuple:
    """Return the class edges a layer's symbology carries, from the one place they could portably live.

    Args:
        symbology: The layer's recorded `Symbology`.

    Returns:
        The `breaks` of the :class:`~digitalearth.base.spec.scale.Scale` on the layer's `color` encoding,
        or `()` when the layer publishes no colour encoding, or one with no scale. Only that one place is
        read: a tier's own spelling of the same edges — MapLibre's `step` expression, HoloViews'
        `color_levels` — is what `Symbology.encodings` exists to replace, so finding the edges there would
        say the figure carries them when only that tier can read them.
    """
    encoding = dict(symbology.encodings).get("color")
    scale = getattr(encoding, "scale", None)
    return tuple(getattr(scale, "breaks", None) or ())


def _described(figure) -> tuple:
    """Reduce a figure to the part of it every tier must describe alike.

    Args:
        figure: The tier's `figure_spec`.

    Returns:
        `(kind, band, visible)` per layer, in draw order. The band is the layer's own override where it set
        one and its kind's band otherwise, because a tier that leaves `band` unset is not thereby drawing
        somewhere else — both tiers here leave it unset and rely on the kind, and comparing the raw field
        would say they agree while saying nothing.

        Ids are left out for the reason the D-11 seam above records. Style is left out because the seed asks
        for none, and a tier's *default* marker size is its own business —
        :class:`TestOneStyledLayerDescribesOneChannelOnBothTiers` is where an explicitly styled layer is
        held to one answer.
    """
    return tuple(
        (layer.kind, layer.band or band_of(layer.kind), layer.visible)
        for layer in figure.layers
    )


def _class_of(backend: str) -> str:
    """Return the name of the test class that signs the contract for a tier.

    Args:
        backend: The tier's name.

    Returns:
        The class name, derived from the backend the same way the classes below are named, so a tier added
        to :data:`TIER_JOBS` without a class is reported as collecting nothing rather than skipped silently.
    """
    return f"Test{backend.capitalize()}MapConformance"


def _collected_ids(target: str) -> tuple:
    """Return the pytest node ids a real collection of this module reports.

    Runs pytest in a child process rather than reasoning about what pytest *would* collect, because the
    regression being guarded against is precisely a gap between the two: the renderer contract's classes
    looked collected to a reader and reported nothing to a CI job.

    Args:
        target: What to collect, as a pytest argument — this module, optionally narrowed to one class.

    Returns:
        The node ids, one per collected item.

    Raises:
        AssertionError: when the child process fails for a reason other than collecting nothing, since a
            crashed collector answering "no items" would otherwise read as a tier that contributes none.
    """
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            target,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode in (0, 5), (
        f"collecting {target!r} failed:\n{finished.stdout}\n{finished.stderr}"
    )
    return tuple(
        line.strip()
        for line in finished.stdout.splitlines()
        if "::" in line and line.strip().startswith(MODULE_PATH)
    )


class MapConformanceBase:
    """The behavioural probes themselves. A tier inherits them by naming itself and handing back a map.

    Every probe asks a question a caller would ask, through the Core names the contract froze — never through
    a tier's own spelling, and never about a signature, which the name half already covers. A tier that
    cannot answer one of them declares that in the module-level table the probe reads, with the reason and a
    guard that fails once the tier can.

    Attributes:
        backend: The tier's name as its `Capabilities` spells it, and the key into every table above.
    """

    backend: str = ""

    def make(self):
        """Return an open map with nothing drawn on it yet.

        Returns:
            The tier's map object.

        Raises:
            NotImplementedError: when the tier has not supplied one.
        """
        raise NotImplementedError

    def close(self, drawn) -> None:
        """Release the map and whatever it registered.

        Args:
            drawn: The object :meth:`make` returned.
        """
        drawn.close()

    @pytest.fixture
    def drawn(self):
        """Yield an open map, closed on the way out.

        The object registry holds strong references to what a builder was handed, so a probe that drew and
        walked away would leak its features for the life of the session.

        Yields:
            The tier's map object.
        """
        built = self.make()
        yield built
        try:
            self.close(built)
        except Exception:  # pragma: no cover - a closed map may refuse a second close
            pass

    def _seed(self, drawn):
        """Draw the shared seed figure — the one :data:`EXPECTED_SEED` describes.

        Points first, then a choropleth, then a graticule. The order is chosen so the result answers two
        questions at once: two data layers keep the order they were added in, and the reference layer added
        last is still described first.

        Args:
            drawn: The map to draw on.

        Returns:
            The map's `figure_spec` after the three calls.
        """
        drawn.points(_points())
        drawn.choropleth(_polygons(), COLUMN, scheme=SCHEME, k=CLASSES)
        drawn.graticule(lon_step=30.0, lat_step=30.0)
        return drawn.figure_spec

    def _hidden_layer(self, drawn):
        """Draw one layer the caller asked to be hidden, and return how the figure describes it.

        Args:
            drawn: The map to draw on.

        Returns:
            The drawn layer's `visible` flag, as the figure reports it.
        """
        drawn.points(_points(), visible=False)
        figure = drawn.figure_spec
        return figure.layers.get(figure.layers.ids[-1]).visible

    def test_two_layers_are_described_in_the_order_they_were_added(self, drawn):
        """Draw two things and the figure should say two things, in the order they were drawn.

        Args:
            drawn: The map under test.

        Test scenario:
            The most basic promise a description makes, and the one a tier can break without any name test
            noticing: three of the four tiers addressed a layer by its position in a list, so an underlay
            inserted at index 0 moved every position a caller held. Two layers of one kind, so the answer is
            about order alone and not about how a tier sorts bands.
        """
        drawn.points(_points())
        drawn.points(_points())
        figure = drawn.figure_spec
        assert [layer.kind for layer in figure.layers] == [POINTS, POINTS], (
            f"the {self.backend} tier drew two point layers and described "
            f"{[layer.kind for layer in figure.layers]}"
        )
        assert tuple(figure.layers.ids) == tuple(drawn.layer_ids), (
            f"the {self.backend} tier's figure and its layer_ids disagree about draw order: "
            f"{tuple(figure.layers.ids)} vs {tuple(drawn.layer_ids)}"
        )

    def test_the_name_a_caller_gives_is_the_id_the_figure_describes(self, drawn):
        """Name a layer and the figure should call it that, on any tier.

        Args:
            drawn: The map under test.

        Test scenario:
            The probe this module reserved and could not write (#321). An id is not decoration — it is what
            `get_layer`, `remove_layer` and a layer switcher key on — so a tier that accepts a name and files
            the layer under a generated one hands the caller an id they never asked for and cannot guess.
            The web and 3-D tiers honoured a name; the static and interactive tiers took no `name=` at all
            and dropped it into the engine's styling, where it either did nothing or coloured something.
        """
        drawn.points(_points(), name=ASKED_NAME)
        assert drawn.figure_spec.layers.ids == (ASKED_NAME,), (
            f"the {self.backend} tier was asked for a layer called {ASKED_NAME!r} and described "
            f"{drawn.figure_spec.layers.ids}"
        )

    def test_a_name_used_twice_is_suffixed_rather_than_shared(self, drawn):
        """Two layers under one name get one id each — the second suffixed, never the same.

        Args:
            drawn: The map under test.

        Test scenario:
            The other half of naming, and the half the tiers disagreed about: `free_layer_id` counts the
            **name** (`wells`, `wells-2`), while the static and interactive tiers counted the *scene* and
            produced `wells-4` when three other layers had been drawn first. Held here rather than per tier
            because the id is what a caller writes down afterwards, so the same script must address the same
            layer whichever backend drew it. Sharing one id instead would be worse than either: it makes the
            second layer unaddressable, and silently drops one of the two from anything keyed by id.
        """
        drawn.points(_points(), name=ASKED_NAME)
        drawn.points(_points(), name=ASKED_NAME)
        assert drawn.figure_spec.layers.ids == (ASKED_NAME, SUFFIXED_NAME), (
            f"the {self.backend} tier drew {ASKED_NAME!r} twice and described "
            f"{drawn.figure_spec.layers.ids}"
        )

    def test_a_reference_layer_is_described_beneath_the_data_it_references(self, drawn):
        """A graticule asked for last is still drawn under the data, so it must be described first.

        Args:
            drawn: The map under test.

        Test scenario:
            Draw order is the band's, not the call's. A tier that appended in call order would put a
            graticule over the map it references — which is what every tier did before the layer tree owned
            z-order, and what a name test cannot see.
        """
        drawn.points(_points())
        drawn.graticule(lon_step=30.0, lat_step=30.0)
        described = [layer.kind for layer in drawn.figure_spec.layers]
        assert described == [GRATICULE, POINTS], (
            f"the {self.backend} tier described {described}; a graticule belongs under the data whenever it "
            "was added"
        )

    def test_the_same_seed_figure_is_described_the_same_way_on_every_tier(self, drawn):
        """The round-trip probe: one sequence of Core calls, one described result, whichever tier made it.

        Args:
            drawn: The map under test.

        Test scenario:
            This is the check `to_backend()` (order 33) will be held to, exercised today against the tier's
            own `figure_spec`. Both tiers are compared with one constant rather than with each other,
            because the `web` and `interactive` jobs each carry one engine and collect one tier — so this
            probe still holds them to each other in the jobs where only one of them is live. The `all`
            environment carries both, and :class:`TestOneStyledLayerDescribesOneChannelOnBothTiers` is what
            asks them directly there.
        """
        described = _described(self._seed(drawn))
        assert described == EXPECTED_SEED, (
            f"the {self.backend} tier describes the seed figure as {described}"
        )

    def test_the_same_classification_gives_the_same_class_edges_on_every_tier(
        self, drawn
    ):
        """One column, one scheme, one class count — one set of breaks on the **live map**, on any tier.

        Args:
            drawn: The map under test.

        Test scenario:
            Both tiers route a graduated `choropleth` through `Scale`, which is the single place the
            classifier is reached, and both publish the result as `last_breaks` — the interactive tier's
            docstring says so explicitly, "for legend parity with the web tier", and nothing checked it. A
            tier that classified locally would draw a legend that disagreed with its own fill.

            `last_breaks` is an attribute of the map object and **no `FigureSpec` carries it** (measured:
            it appears in no layer's `symbology.props` and in no `encodings` on either tier). So this is
            live-map parity between two engines, not a statement about what survives a write to JSON, and
            it is not part of the `to_backend()` precondition the probes above are — a probe cannot be the
            precondition for carrying a figure across while reading something the figure does not carry
            (review R-M6). What the figure carries is asked separately, by
            :meth:`test_a_tier_whose_figure_carries_no_class_edges_is_named_as_such`.
        """
        drawn.choropleth(_polygons(), COLUMN, scheme=SCHEME, k=CLASSES)
        assert tuple(drawn.last_breaks) == EXPECTED_BREAKS, (
            f"the {self.backend} tier cut {COLUMN!r} into {tuple(drawn.last_breaks)}"
        )

    def test_a_tier_whose_figure_carries_no_class_edges_is_named_as_such(self, drawn):
        """And the figure's own answer to the same question, which is a different answer.

        Args:
            drawn: The map under test.

        Test scenario:
            A drift guard over the gap the probe above leaves, in both directions.
            :class:`~digitalearth.base.spec.scale.Scale` on the layer's `color` encoding is where class
            edges would have to live for `to_backend()` to carry them, and :func:`_class_edges` looks
            exactly there. A tier on :data:`NO_PORTABLE_CLASSIFICATION` must carry none; a tier off it must
            carry :data:`EXPECTED_BREAKS`, so the day a tier publishes its classification the entry comes
            off and this becomes the cross-seam check the classification probe was mistaken for.
        """
        drawn.choropleth(_polygons(), COLUMN, scheme=SCHEME, k=CLASSES)
        figure = drawn.figure_spec
        symbology = figure.layers.get(figure.layers.ids[-1]).symbology
        carried = _class_edges(symbology)
        if self.backend in NO_PORTABLE_CLASSIFICATION:
            assert carried == (), (
                f"the {self.backend} tier's figure now carries {carried}; take it off "
                "NO_PORTABLE_CLASSIFICATION so the cross-tier check holds it"
            )
            return
        assert carried == EXPECTED_BREAKS, (
            f"the {self.backend} tier's figure carries {carried} for a graduated choropleth; publish the "
            "Scale on the colour encoding, or name the tier in NO_PORTABLE_CLASSIFICATION"
        )

    def test_a_layer_nobody_hid_is_described_visible(self, drawn):
        """The complement that stops the hidden-layer probes passing by describing everything hidden.

        Args:
            drawn: The map under test.
        """
        drawn.points(_points())
        figure = drawn.figure_spec
        assert figure.layers.get(figure.layers.ids[-1]).visible is True, (
            f"the {self.backend} tier described a layer nobody hid as hidden"
        )

    def test_a_layer_built_hidden_is_described_hidden(self, drawn):
        """Ask for a layer to start hidden and the figure should say it is hidden.

        Args:
            drawn: The map under test.

        Test scenario:
            A switcher reads the figure to decide which boxes are ticked. A tier that accepts `visible=False`
            and describes the layer visible hands the switcher a tick for a layer nobody can see — and, where
            the flag reached the engine anyway, the description and the drawing disagree about one layer.
            The tiers that do exactly that are named in :data:`CANNOT_HIDE_AT_BUILD` and skip here.
        """
        excuse = CANNOT_HIDE_AT_BUILD.get(self.backend)
        if excuse is not None:
            pytest.skip(
                f"the {self.backend} tier cannot build a hidden layer: {excuse}"
            )
        assert self._hidden_layer(drawn) is False, (
            f"the {self.backend} tier was asked for a hidden layer and described it visible"
        )

    def test_a_tier_on_the_hidden_at_build_list_still_belongs_there(self, drawn):
        """An excuse must not outlive what it excuses, so the list is checked against the tier.

        Args:
            drawn: The map under test.

        Test scenario:
            The half of the allowlist pattern that keeps it honest. A tier named in
            :data:`CANNOT_HIDE_AT_BUILD` is asked for a hidden layer here too; the day it describes one
            correctly, this fails and the entry has to come off — which is what turns the probe above on for
            that tier. Without it the list would quietly outlast the defect, and the probe would stay off
            forever.
        """
        if self.backend not in CANNOT_HIDE_AT_BUILD:
            pytest.skip(f"the {self.backend} tier is not on the hidden-at-build list")
        assert self._hidden_layer(drawn) is True, (
            f"the {self.backend} tier now describes a layer built hidden as hidden; take it off "
            "CANNOT_HIDE_AT_BUILD so its probe runs"
        )

    def test_a_tier_that_records_no_portable_channel_is_named_as_such(self, drawn):
        """`Symbology.encodings` is the one style reading any tier can take from any other.

        Args:
            drawn: The map under test.

        Test scenario:
            A drift guard over the finding, in both directions. The layer is drawn with an explicit `size=`,
            which the contract declares as a keyword of `points` and `CHANNELS` declares as a channel, so
            there really is something to record. A tier not on :data:`NO_PORTABLE_CHANNELS` must record it,
            **under that channel and with the value asked for** — recording *something* would pass a tier
            that filed the number under the wrong channel; a tier on the list must still record nothing.
        """
        drawn.points(_points(), size=SIZE)
        figure = drawn.figure_spec
        symbology = figure.layers.get(figure.layers.ids[-1]).symbology
        recorded = sorted(symbology.encodings)
        if self.backend in NO_PORTABLE_CHANNELS:
            assert recorded == [], (
                f"the {self.backend} tier now records {recorded}; take it off NO_PORTABLE_CHANNELS"
            )
            return
        assert "size" in recorded, (
            f"the {self.backend} tier records {recorded} for a layer drawn with size=; fold its style into "
            "Symbology.encodings, or name it in NO_PORTABLE_CHANNELS with what it records instead"
        )
        assert symbology.encoding("size").resolve() == SIZE, (
            f"the {self.backend} tier was asked for size={SIZE} and records "
            f"{symbology.encoding('size').resolve()!r}"
        )

    def test_an_unstyled_layer_publishes_no_portable_channel(self, drawn):
        """A layer nobody styled must publish no caller intent, because there was none to publish.

        Args:
            drawn: The map under test.

        Test scenario:
            The other half of the probe above, and the one the module docstring already promised: *"Which
            default a tier picks is not what a round trip has to agree about; what a caller **asked** for
            is."* `Symbology.encodings` is the field `to_backend()` (order 33) will read, so a tier that
            lifts its own builder defaults into it publishes them **as** the caller's intent — and the two
            tiers' defaults differ, so carrying an unstyled web layer across would draw it at the web
            tier's colour, opacity and size rather than the target tier's own (review R-H2). Measured
            before the fix: the web tier published `{'color': '#3388ff', 'opacity': 0.9, 'size': 5.0}` and
            the interactive tier `{'size': 6.0}` for the same `points(features)`.
        """
        drawn.points(_points())
        figure = drawn.figure_spec
        symbology = figure.layers.get(figure.layers.ids[-1]).symbology
        published = {
            channel: symbology.encoding(channel).resolve()
            for channel in sorted(symbology.encodings)
        }
        assert published == {}, (
            f"the {self.backend} tier was asked for an unstyled layer and published {published} as the "
            "caller's own style; a value its builder defaulted to is the tier's business, not a portable ask"
        )

    def test_this_tier_contributes_a_non_zero_count_of_probes(self):
        """A tier whose class is live must really bring every probe with it.

        Test scenario:
            The per-tier half of the collection guard. It runs only where the tier's engine is installed —
            that is what its class is gated on — so reaching it at all is the evidence that this environment
            collects and runs this module for this tier. It then counts what the tier contributes, because a
            subclass that shadowed the probes, or a base class that lost them, would leave a green job
            covering nothing.

            The relation is **containment, not equality**: a tier must bring every shared probe, and may
            bring probes of its own. Equality made any tier-specific probe a failure, reported as a
            "shadowed" probe, so a tier could never ask a question only it can answer (review R-L8).
            Containment alone would miss a subclass that keeps a probe's name and empties it — `dir()`
            reports the name either way — so each declared probe is also asked whether it is still
            callable here.
        """
        mine = {name for name in dir(self) if name.startswith("test_")}
        declared = sorted(
            name for name in vars(MapConformanceBase) if name.startswith("test_")
        )
        missing = sorted(set(declared) - mine)
        assert missing == [], (
            f"the {self.backend} tier brings {len(mine)} probes and is missing {len(missing)} of the "
            f"{len(declared)} the contract declares: {missing}"
        )
        assert declared != [], "the contract declares no probes at all"
        emptied = sorted(
            name for name in declared if not callable(getattr(self, name, None))
        )
        assert emptied == [], (
            f"the {self.backend} tier shadows {emptied} with something that is not callable, so those "
            "probes are collected and cover nothing"
        )


#: Skipping is per tier, not per module. A module-level `importorskip` would skip the base class and the
#: collection guard as well — in an environment where they are exactly what needs to run.
needs_maplibre = pytest.mark.skipif(
    importlib.util.find_spec("maplibre") is None,
    reason="the web tier needs the web environment",
)

needs_geoviews = pytest.mark.skipif(
    importlib.util.find_spec("geoviews") is None,
    reason="the interactive tier needs the interactive environment",
)


@needs_maplibre
class TestWebMapConformance(MapConformanceBase):
    """The web (MapLibre + deck.gl) tier."""

    backend = "web"

    def make(self):
        """Return an empty web map.

        Returns:
            The map. Constructing one touches no engine; the builders lazy-import MapLibre.
        """
        from digitalearth.web import WebMap

        return WebMap()


@needs_geoviews
class TestInteractiveMapConformance(MapConformanceBase):
    """The interactive (HoloViz) tier."""

    backend = "interactive"

    def make(self):
        """Return an empty interactive map.

        Returns:
            The map. Constructing one touches no engine; the builders lazy-import HoloViz.
        """
        from digitalearth.interactive import InteractiveMap

        return InteractiveMap()


class TestEveryTierIsReallyCollected:
    """The guard the renderer contract did not have, and paid for (review M6).

    Its classes were gated on optional engines and the only job collecting `tests/base` had neither
    installed, so every check it declared skipped, in every job, for its whole life — and the suite was
    green throughout. Three checks close that here. The first two read the manifest and run in **every**
    environment, so they cannot themselves go uncollected; the third runs a real collection and counts.
    """

    @staticmethod
    def _manifest() -> dict:
        """Return the parsed `pyproject.toml`.

        Returns:
            The manifest, as a dict.
        """
        return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    @pytest.mark.parametrize("backend", sorted(TIER_JOBS))
    def test_the_tiers_job_collects_this_module(self, backend):
        """A tier's task must name this module, or the job runs none of its probes.

        Args:
            backend: The tier under test.

        Test scenario:
            `test-web` collects `tests/web` and `test-interactive` collects `tests/interactive`; neither
            reaches `tests/base` on its own. That is exactly how the renderer contract went uncollected, and
            it was fixed the same way — by naming the module in the task's own command.
        """
        task, _, _ = TIER_JOBS[backend]
        defined = self._manifest()["tool"]["pixi"]["tasks"][task]
        command = defined["cmd"] if isinstance(defined, dict) else defined
        assert MODULE_PATH in command, (
            f"the {task!r} task does not collect {MODULE_PATH}, so the {backend} tier's probes never run: "
            f"{command}"
        )

    @pytest.mark.parametrize("backend", sorted(TIER_JOBS))
    def test_the_tiers_environment_carries_its_engine(self, backend):
        """Collecting the module is half of it; the environment must also ungate the tier's class.

        Args:
            backend: The tier under test.

        Test scenario:
            The other half of the same failure. A job that collects this module in an environment without
            the engine collects the tier's class and skips every item in it — which reads green and covers
            nothing, and is what the `main` matrix did to the renderer contract.
        """
        _, environment, feature = TIER_JOBS[backend]
        features = self._manifest()["tool"]["pixi"]["environments"][environment][
            "features"
        ]
        assert feature in features, (
            f"the {environment!r} environment does not carry the {feature!r} feature, so the {backend} "
            f"tier's class would skip there; it has {features}"
        )

    def test_an_installed_tier_collects_a_non_zero_item_count(self):
        """Where a tier's engine is importable, collecting this module must report items for it.

        Test scenario:
            The measured half. `--collect-only` is run in a child process against this module, and every
            tier whose engine this environment has must appear in the result. An environment with no tier
            engine at all — `dev`, `viz3d`, the pure-wheel job — skips rather than passing on an empty
            question, and spawns no child process for an answer it could not check; the two manifest checks
            above are what cover those environments, which is why they read the manifest instead.
        """
        installed = [
            backend
            for backend, engine in sorted(TIER_ENGINES.items())
            if importlib.util.find_spec(engine) is not None
        ]
        if not installed:
            pytest.skip(
                "no tier engine is installed here, so there is no item count to measure; the two manifest "
                "checks above are what hold this environment"
            )
        collected = _collected_ids(MODULE_PATH)
        counted = {
            backend: len(
                [node for node in collected if f"::{_class_of(backend)}::" in node]
            )
            for backend in installed
        }
        empty = sorted(backend for backend, count in counted.items() if count == 0)
        assert empty == [], (
            f"{empty} have their engine installed here and collected no items from {MODULE_PATH}; "
            f"collected {len(collected)} items in total"
        )


class TestOneStyledLayerDescribesOneChannelOnBothTiers:
    """Two live 2-D tiers, one process, one styled layer — and one answer to "how is this styled?".

    The probes above hold each tier to a **constant**, because until the `all` environment existed no
    interpreter carried MapLibre and HoloViz at once. That is enough to stop the two drifting apart, and it
    is not enough to show that a style *crosses*: a constant can be met by a tier that spells the same
    reading in its own way. This asks the two directly, in one process, which is the capability the `all`
    environment exists for and the precondition `to_backend()` (order 33) is built on — carrying a figure
    across is only worth doing if both ends describe its style in the same words.

    The layer is styled explicitly, in each tier's own keyword spelling (:data:`CROSS_TIER_STYLE`): the web
    tier's opacity keyword is `opacity=` and the interactive tier's is `alpha=` (#332). The promise is not
    that the keywords match — it is that both describe the same *channel*, which is the whole reason a
    channel vocabulary exists.

    The declaration check runs in **every** environment, so this class cannot become the thing this module
    was written to prevent: a contract that reads green because nothing collected it.
    """

    @staticmethod
    def _channels(build, style: dict) -> tuple:
        """Draw one styled point layer and return the channels the figure says it drives.

        Args:
            build: A no-argument callable returning an empty map of one tier.
            style: The keywords to draw with, in that tier's own spelling.

        Returns:
            `(channel, value)` per described encoding, sorted by channel — the portable reading of the
            layer's style, with no tier name anywhere in it.
        """
        drawn = build()
        try:
            drawn.points(_points(), **style)
            figure = drawn.figure_spec
            symbology = figure.layers.get(figure.layers.ids[-1]).symbology
            return tuple(
                (channel, symbology.encoding(channel).resolve())
                for channel in sorted(symbology.encodings)
            )
        finally:
            try:
                drawn.close()
            except (
                Exception
            ):  # pragma: no cover - a closed map may refuse a second close
                pass

    @staticmethod
    def _web() -> tuple:
        """Return the web tier's reading of the shared styled layer.

        Returns:
            Its `(channel, value)` pairs.
        """
        from digitalearth.web import WebMap

        return TestOneStyledLayerDescribesOneChannelOnBothTiers._channels(
            WebMap, CROSS_TIER_STYLE["web"]
        )

    @staticmethod
    def _interactive() -> tuple:
        """Return the interactive tier's reading of the shared styled layer.

        Returns:
            Its `(channel, value)` pairs.
        """
        from digitalearth.interactive import InteractiveMap

        return TestOneStyledLayerDescribesOneChannelOnBothTiers._channels(
            InteractiveMap, CROSS_TIER_STYLE["interactive"]
        )

    def test_the_cross_backend_job_collects_this_module(self):
        """The environment with both engines must run this module, or nothing here ever executes.

        Test scenario:
            The manifest half, and the reason it is here rather than assumed: this class skips in every
            single-backend job by design, so the only evidence that it runs at all is that the one job which
            can run it names this module. Read from `pyproject.toml`, so it answers in every environment —
            including the lean `dev` matrix, where neither engine is installed.
        """
        task, environment = CROSS_TIER_JOB
        manifest = tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        defined = manifest["tool"]["pixi"]["tasks"][task]
        command = defined["cmd"] if isinstance(defined, dict) else defined
        assert MODULE_PATH in command, (
            f"the {task!r} task does not collect {MODULE_PATH}, so the cross-tier probes never run: "
            f"{command}"
        )
        carried = manifest["tool"]["pixi"]["environments"][environment]["features"]
        missing = sorted(
            feature for _, _, feature in TIER_JOBS.values() if feature not in carried
        )
        assert missing == [], (
            f"the {environment!r} environment is missing {missing}, so it cannot hold both 2-D tiers in "
            f"one process; it carries {carried}"
        )

    @needs_maplibre
    def test_the_web_tier_describes_the_styled_layer_by_channel(self):
        """One half of the pair, held to the constant so a wrong pair cannot agree its way to green."""
        assert self._web() == EXPECTED_CHANNELS, (
            f"the web tier describes the styled layer as {self._web()}"
        )

    @needs_geoviews
    def test_the_interactive_tier_describes_the_styled_layer_by_channel(self):
        """The other half, held to the same constant, through its own keyword spelling."""
        assert self._interactive() == EXPECTED_CHANNELS, (
            f"the interactive tier describes the styled layer as {self._interactive()}"
        )

    @needs_maplibre
    @needs_geoviews
    def test_both_tiers_describe_one_styled_layer_by_the_same_channels(self):
        """The check no single-backend job can run: two live tiers, one interpreter, one style.

        Test scenario:
            Each side is built from a different tier and a different keyword spelling, so the two are not
            the same expression reaching the same answer — they are two engines asked the same question.
            Held against each other *as well as* against the constant above, because the constant catches a
            tier that drifts and this catches the pair drifting together.
        """
        assert self._web() == self._interactive(), (
            f"the web tier describes {self._web()} and the interactive tier {self._interactive()}"
        )


class TestTheTablesDescribeThePackage:
    """Every tier is either held by the probes above or named in a table, and nothing is silently absent."""

    def test_every_tier_either_signs_the_contract_or_is_named_in_a_table(self):
        """A tier that neither subclasses the base nor appears in a drift table is covered by nothing.

        Test scenario:
            The module opens "One set of behavioural questions **every tier's** map answers the same way",
            and two of the four tiers subclass it. That is defensible — and it was said nowhere, while
            `NO_PORTABLE_CHANNELS` sat empty beside two tiers that record no portable channel at all, so
            the ledger read as a statement about the package and was one about the subset (review R-M4).
        """
        unheld = sorted(
            backend
            for backend in ALL_TIERS
            if backend not in TIER_JOBS and backend not in NO_PORTABLE_CHANNELS
        )
        assert unheld == [], (
            f"{unheld} neither subclass MapConformanceBase nor appear in NO_PORTABLE_CHANNELS, so nothing "
            "in this module says anything about them"
        )

    def test_every_tier_a_table_excuses_can_be_re_asked_the_question(self):
        """An excuse is only honest while something re-asks the question it excuses.

        Test scenario:
            A tier named in a table is either a subclass — re-asked by the base class's reverse probes — or
            reachable through :data:`ABSENT_TIERS`, where the two checks below re-ask it. A name in neither
            place would be an excuse nothing can contradict, which is the shape this module exists to
            refuse.
        """
        named = (
            set(CANNOT_HIDE_AT_BUILD)
            | set(NO_PORTABLE_CHANNELS)
            | set(NO_PORTABLE_CLASSIFICATION)
        )
        unreachable = sorted(named - set(TIER_JOBS) - set(ABSENT_TIERS))
        assert unreachable == [], (
            f"{unreachable} are excused by a table and reachable from nowhere; give each one a row in "
            "ABSENT_TIERS or a subclass, so the day it is fixed something says so"
        )

    @pytest.mark.parametrize("backend", sorted(ABSENT_TIERS))
    def test_a_tier_excused_from_portable_channels_still_records_none(self, backend):
        """The reverse branch for the tiers that do not subclass the base.

        Args:
            backend: The tier under test.

        Test scenario:
            The same promise :meth:`MapConformanceBase.test_a_tier_that_records_no_portable_channel_is_named_as_such`
            makes for the subclassed tiers, asked of the two that cannot subclass. The layer is drawn with
            an explicit `size=`, which `CHANNELS` declares, so a tier that grew a fold would record
            something here and the entry would have to come off — which is what turns a real probe on for
            it. Before this existed the branch was unreachable in both directions (review R-L7).
        """
        if backend not in NO_PORTABLE_CHANNELS:
            pytest.skip(
                f"the {backend} tier is not excused from recording portable channels"
            )
        tier = ABSENT_TIERS[backend]
        pytest.importorskip(tier.engine)
        drawn = tier.make()
        try:
            tier.styled(drawn)
            figure = drawn.figure_spec
            symbology = figure.layers.get(figure.layers.ids[-1]).symbology
            recorded = sorted(symbology.encodings)
        finally:
            _closed(drawn)
        assert recorded == [], (
            f"the {backend} tier now records {recorded} for a layer drawn with size={SIZE}; take it off "
            "NO_PORTABLE_CHANNELS and give it a MapConformanceBase subclass, so the probes hold it"
        )

    @pytest.mark.parametrize("backend", sorted(ABSENT_TIERS))
    def test_a_tier_excused_from_hiding_at_build_still_cannot(self, backend):
        """The other reverse branch, for the same two tiers.

        Args:
            backend: The tier under test.

        Test scenario:
            A tier on :data:`CANNOT_HIDE_AT_BUILD` is asked for a hidden layer here too; the day it
            describes one correctly, this fails and the entry has to come off. `"refused"` and `"visible"`
            are both failures of the promise and so both leave the entry standing — only a layer really
            described hidden retires it.
        """
        if backend not in CANNOT_HIDE_AT_BUILD:
            pytest.skip(f"the {backend} tier is not on the hidden-at-build list")
        tier = ABSENT_TIERS[backend]
        pytest.importorskip(tier.engine)
        drawn = tier.make()
        try:
            outcome = _asked_to_hide(tier, drawn)
        finally:
            _closed(drawn)
        assert outcome != "hidden", (
            f"the {backend} tier now describes a layer built hidden as hidden; take it off "
            "CANNOT_HIDE_AT_BUILD so its probe runs"
        )


class TestATierMayBringAProbeOfItsOwn:
    """The count check must let a tier add to the contract, and still refuse a tier that subtracts."""

    @staticmethod
    def _with_its_own():
        """Return a subclass that declares one probe the base class does not.

        Returns:
            The class. Its name does not begin with `Test`, so pytest collects nothing from it — the point
            is the count check's answer about it, not running its probes.
        """

        class _WithItsOwn(MapConformanceBase):
            """A tier that answers every shared question and one of its own."""

            backend = "make-believe"

            def test_something_only_this_tier_can_answer(self):
                """A probe a single tier is entitled to add, which is what the check must tolerate."""

        return _WithItsOwn

    def test_a_subclass_that_adds_a_probe_still_satisfies_the_count_check(self):
        """A tier-specific probe is a contribution, not a defect.

        Test scenario:
            The check asserted set **equality** between what a tier contributes and what the base declares,
            so any probe a tier added made the two differ and failed — with a message about "shadowed"
            probes that would not have described what happened. A tier could therefore never bring a
            question only it can answer (review R-L8).
        """
        self._with_its_own()().test_this_tier_contributes_a_non_zero_count_of_probes()

    def test_a_subclass_that_shadows_a_probe_with_a_non_callable_is_still_refused(self):
        """The direction the check exists for has to survive the widening.

        Test scenario:
            Relaxing equality to "declares at least the shared probes" is only safe while something still
            catches a subclass that keeps a probe's *name* and empties it — `dir()` reports the name either
            way, so a name check alone would pass. The check asks whether each declared probe is still
            callable on the tier.
        """
        shadowing = self._with_its_own()
        shadowing.test_a_layer_nobody_hid_is_described_visible = None
        with pytest.raises(AssertionError) as refusal:
            shadowing().test_this_tier_contributes_a_non_zero_count_of_probes()
        assert "test_a_layer_nobody_hid_is_described_visible" in str(refusal.value)
