"""Structural guards for the backend mixin contract and the fluent-builder return type (#172).

Every capability mixin in ``static/``, ``interactive/``, ``three_d/`` and ``web/`` declares its backend's base
class as its own base **under ``TYPE_CHECKING`` only**::

    if TYPE_CHECKING:
        from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
    else:
        _MixinBase = object

    class VectorMixin(_MixinBase):
        ...

That gives mypy the composed-class contract the mixin's methods reach through ``self`` while leaving the runtime
class hierarchy exactly as it was. The whole construction is only safe as long as the ``else`` branch keeps
winning at runtime, and the failure mode is silent in the one backend where it matters most: if the guard were
dropped, ``digitalearth.interactive``/``web``/``three_d`` would raise a *loud* MRO ``TypeError`` at import, but
``static.map.Map`` lists ``GeoLayerBase`` last, so the MRO would still linearise to the very same nine class **names**
that ``tests/static/test_map_composition.py`` pins — while every mixin quietly gained a real base class, and with it a
different ``super()`` chain and a different ``__init__``. Nothing else in the suite compares ``__bases__``, so
this file does.

The second guard is on the PEP 673 ``Self`` return type the 67 fluent builders now carry in place of the mixin
name they used to quote. ``-> Self`` is a promise that the call returns the *composed* map, so builders chain;
mypy cannot enforce it everywhere because ``no-any-return`` is baselined off in several of these modules, so the
promise is checked here instead: every ``-> Self`` method must return ``self``, or delegate to another ``-> Self``
method that does.

The third guard reads the contract the other way round: a method that does a builder's work must not
declare a builder's opposite. Its reach is measured rather than claimed — 40 classes and 265 public
methods, being every ``*Mixin`` in the four backend packages **plus** the base and composed classes they
compose into (``Scene``, ``GeoLayerBase``, ``Map``, ``TexturedGlobe`` and the three other tiers' pairs) —
and it tests "not ``Self``" rather than "``-> None``", so a missing annotation, the string ``'None'``, an
``-> Any`` artist return and a registrar reached through a private helper or a closure all read as the same
finding. The eleven methods that deliberately answer something else are enumerated, with a reason each, in
:data:`CONTRACT_CARVE_OUTS`. For contrast, the version this replaced read 30 classes and 3 methods.

Every scan here reads the source with :mod:`ast` rather than importing, so all 27 mixins are covered in
every environment — including the five ``three_d`` modules that need PyVista, which the ``dev`` env does not
have. The runtime assertions are the belt to that braces, and skip where an optional engine is missing.
"""

import ast
import functools
import importlib
import pathlib
import textwrap

import pytest

#: Repository root — this file lives in `<root>/tests/`.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: `<backend package> -> (module holding its base class, base class name)`. The mixins in each package must
#: declare that base under TYPE_CHECKING; the composed class in the same package inherits it for real.
BACKENDS = {
    "digitalearth.static.maps": ("digitalearth.static.maps.base", "GeoLayerBase"),
    "digitalearth.interactive": ("digitalearth.interactive.base", "InteractiveMapBase"),
    "digitalearth.three_d": ("digitalearth.three_d.base", "Scene3DBase"),
    "digitalearth.web": ("digitalearth.web.base", "WebMapBase"),
}

#: How many mixins the four backends held when this guard was written. A floor, not an equality: adding a
#: capability mixin is normal, and the scan picks it up automatically. It exists so a broken scan cannot make
#: every set-based assertion below pass over an empty list.
MIXIN_FLOOR = 27

#: How many `-> Self` builders the backends carried at #172. Same role as `MIXIN_FLOOR`.
BUILDER_FLOOR = 60

#: Third-party engines a backend module may fail to import when its extra is not installed. Only these turn an
#: import failure into a skip — a `ModuleNotFoundError` naming anything else is a real break and must fail.
OPTIONAL_ENGINES = {
    "datashader",
    "geoviews",
    "geovista",
    "holoviews",
    "lonboard",
    "maplibre",
    "panel",
    "pyvista",
}


def _module_path(module: str) -> pathlib.Path:
    """Return the file backing a dotted module name under `src/`."""
    return ROOT / "src" / pathlib.Path(*module.split("."))


@functools.cache
def _mixins() -> tuple:
    """Return `(backend, module, class name, ClassDef, Module)` for every `*Mixin` in the four backends.

    Read straight off the source with `ast`, so a mixin whose engine is not installed is still covered.
    """
    found = []
    for backend in BACKENDS:
        for path in sorted(_module_path(backend).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name.endswith("Mixin"):
                    module = f"{backend}.{path.stem}"
                    found.append((backend, module, node.name, node, tree))
    return tuple(found)


def _mixin_ids() -> list:
    """Return `module:ClassName` ids for parametrising over every discovered mixin."""
    return [f"{module}:{name}" for _, module, name, _, _ in _mixins()]


def _runtime_fallback(tree: ast.Module) -> list:
    """Return the values assigned to `_MixinBase` in the `else` branch of an `if TYPE_CHECKING:` block."""
    assigned = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name)):
            continue
        if node.test.id != "TYPE_CHECKING":
            continue
        for statement in node.orelse:
            if isinstance(statement, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_MixinBase"
                for t in statement.targets
            ):
                assigned.append(ast.unparse(statement.value))
    return assigned


def _type_checking_import(tree: ast.Module) -> list:
    """Return `(module, name)` for every symbol imported as `_MixinBase` under `if TYPE_CHECKING:`."""
    imported = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name)):
            continue
        if node.test.id != "TYPE_CHECKING":
            continue
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    if alias.asname == "_MixinBase":
                        imported.append((statement.module, alias.name))
    return imported


def _import_or_skip(module: str):
    """Import `module`, or skip the test when the optional engine it needs is not installed.

    Args:
        module: Dotted module name to import.

    Returns:
        The imported module.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name in OPTIONAL_ENGINES:
            pytest.skip(f"{module} needs the optional engine {exc.name!r}")
        raise


class TestMixinDeclaration:
    """Tests for the `if TYPE_CHECKING` / `else` contract every backend mixin declares."""

    def test_the_scan_finds_every_mixin(self):
        """The source scan finds at least the 27 mixins the four backends shipped at #172.

        Test scenario:
            Every other assertion in this class iterates the scan's output, so an empty or truncated scan
            would satisfy all of them for the wrong reason. Pin a floor, and require all four backends to
            contribute, so a moved directory fails loudly instead of silently disabling the guard.
        """
        found = _mixins()
        backends = {backend for backend, _, _, _, _ in found}
        assert len(found) >= MIXIN_FLOOR, (
            f"expected >={MIXIN_FLOOR} backend mixins, found {len(found)}: {_mixin_ids()}"
        )
        assert backends == set(BACKENDS), (
            f"expected mixins in every backend {sorted(BACKENDS)}, found {sorted(backends)}"
        )

    @pytest.mark.parametrize("index", range(len(_mixins())), ids=_mixin_ids())
    def test_mixin_declares_exactly_the_guarded_base(self, index):
        """Each mixin's only declared base is the `_MixinBase` alias.

        Args:
            index: Position of the mixin in the source scan.

        Test scenario:
            A mixin that inherits its backend base directly (or anything else) would change the composed
            MRO; a mixin that inherits nothing at all would lose the contract mypy checks against.
        """
        _, module, name, node, _ = _mixins()[index]
        bases = [ast.unparse(base) for base in node.bases]
        assert bases == ["_MixinBase"], (
            f"{module}.{name} must declare exactly ['_MixinBase'] as its base, found {bases}"
        )

    @pytest.mark.parametrize("index", range(len(_mixins())), ids=_mixin_ids())
    def test_runtime_fallback_binds_object(self, index):
        """The `else` branch of the `TYPE_CHECKING` guard binds `_MixinBase` to `object`.

        Args:
            index: Position of the mixin in the source scan.

        Test scenario:
            This branch is the whole reason the runtime hierarchy is unchanged. Binding anything else — or
            omitting the branch, which would leave `_MixinBase` undefined — breaks the composition.
        """
        _, module, name, _, tree = _mixins()[index]
        assigned = _runtime_fallback(tree)
        assert assigned == ["object"], (
            f"{module} (home of {name}) must bind `_MixinBase = object` at runtime, found {assigned}"
        )

    @pytest.mark.parametrize("index", range(len(_mixins())), ids=_mixin_ids())
    def test_type_checked_base_is_the_backend_base(self, index):
        """The base imported under `TYPE_CHECKING` is the base class of the mixin's own backend.

        Args:
            index: Position of the mixin in the source scan.

        Test scenario:
            The alias is what mypy checks the mixin's `self.<...>` calls against, so pointing it at another
            backend's base (an easy copy-paste slip between four near-identical guards) would type-check the
            mixin against a contract the composed class never fulfils, with no runtime symptom at all.
        """
        backend, module, name, _, tree = _mixins()[index]
        assert _type_checking_import(tree) == [BACKENDS[backend]], (
            f"{module}.{name} must alias {BACKENDS[backend]} as `_MixinBase`, "
            f"found {_type_checking_import(tree)}"
        )


class TestMixinRuntimeBases:
    """Tests for what the mixins actually inherit once the modules are imported."""

    @pytest.mark.parametrize("index", range(len(_mixins())), ids=_mixin_ids())
    def test_mixin_is_a_plain_object_subclass(self, index):
        """An imported mixin's `__bases__` is `(object,)`.

        Args:
            index: Position of the mixin in the source scan.

        Test scenario:
            The runtime counterpart of the source guard, and the only assertion in the suite that would
            catch the `static` backend regressing: if the mixins inherited `GeoLayerBase` for real, `Map`'s
            MRO would still linearise to the same nine class *names*, so the existing composition test would
            stay green while every mixin had gained a base class and a different `super()` chain.
        """
        _, module, name, _, _ = _mixins()[index]
        mixin = getattr(_import_or_skip(module), name)
        assert mixin.__bases__ == (object,), (
            f"{module}.{name}.__bases__ must be (object,), found {mixin.__bases__}"
        )

    @pytest.mark.parametrize("index", range(len(_mixins())), ids=_mixin_ids())
    def test_module_level_alias_is_object(self, index):
        """The module-level `_MixinBase` name is `object` after import.

        Args:
            index: Position of the mixin in the source scan.

        Test scenario:
            `TYPE_CHECKING` is `False` at runtime, so the alias the class statement consumed must be the
            builtin. Asserting the name as well as the resulting class pins the mechanism, not just its
            effect, which keeps the failure message pointing at the guard that broke.
        """
        _, module, name, _, _ = _mixins()[index]
        imported = _import_or_skip(module)
        assert imported._MixinBase is object, (
            f"{module}._MixinBase must be `object` at runtime (home of {name}), "
            f"found {imported._MixinBase!r}"
        )


class TestComposedClassMro:
    """Tests for the exact method-resolution order of the composed backend classes.

    All four backends list their capability mixins first and their backend base last. That order is load
    bearing rather than stylistic: with the mixins typed against that base (see :mod:`digitalearth.web.base`
    and friends), listing the base *first* puts it both before and after its own subclasses, and the MRO
    algorithm cannot linearize it — mypy then falls back to ``[cls, object]`` and every attribute silently
    becomes ``Any``. ``static.Map`` always had the working order; the other three were corrected to match.
    """

    @pytest.mark.parametrize(
        "module, name, expected",
        [
            (
                "digitalearth.interactive.map",
                "InteractiveMap",
                [
                    "InteractiveMap",
                    "RasterMixin",
                    "VectorMixin",
                    "BigDataMixin",
                    "TemporalMixin",
                    "DecorationMixin",
                    "InteractionMixin",
                    "ProjectionMixin",
                    "AnimationMixin",
                    "DashboardMixin",
                    "InteractiveMapBase",
                    "object",
                ],
            ),
            (
                "digitalearth.web.map",
                "WebMap",
                [
                    "WebMap",
                    "RasterMixin",
                    "VectorMixin",
                    "BigDataMixin",
                    "ThreeDMixin",
                    "TemporalMixin",
                    "DecorationMixin",
                    "ExportMixin",
                    "WebMapBase",
                    "object",
                ],
            ),
            (
                "digitalearth.three_d.scene3d",
                "Scene3D",
                [
                    "Scene3D",
                    "TerrainMixin",
                    "PointCloudMixin",
                    "VolumeMixin",
                    "VectorMixin",
                    "GlobeMixin",
                    "DecorationMixin",
                    "GuideMixin",
                    "AnimationMixin",
                    "Scene3DBase",
                    "object",
                ],
            ),
        ],
        ids=["interactive", "web", "three_d"],
    )
    def test_mro_is_mixins_then_base_then_object(self, module, name, expected):
        """Each composed class linearises to its mixins in declaration order, then its base, then `object`.

        Args:
            module: Module holding the composed class.
            name: The composed class.
            expected: The class names the MRO must contain, in order.

        Test scenario:
            The existing per-backend tests assert only that each mixin appears *somewhere* in the MRO, which
            a re-ordering — or a mixin gaining a real base — would survive. Order decides which definition
            wins when two mixins name the same method, so it is pinned here. `static.Map` is not repeated:
            its MRO is already pinned by `tests/static/test_map_composition.py`.
        """
        composed = getattr(_import_or_skip(module), name)
        assert [cls.__name__ for cls in composed.__mro__] == expected, (
            f"unexpected {name} MRO: {[cls.__name__ for cls in composed.__mro__]}"
        )

    @pytest.mark.parametrize(
        "module, name, backend",
        [
            (
                "digitalearth.interactive.map",
                "InteractiveMap",
                "digitalearth.interactive",
            ),
            ("digitalearth.web.map", "WebMap", "digitalearth.web"),
            ("digitalearth.three_d.scene3d", "Scene3D", "digitalearth.three_d"),
            ("digitalearth.static.map", "Map", "digitalearth.static.maps"),
        ],
        ids=["interactive", "web", "three_d", "static"],
    )
    def test_composed_class_inherits_its_backend_base(self, module, name, backend):
        """The composed class is the only place the backend base is inherited for real.

        Args:
            module: Module holding the composed class.
            name: The composed class.
            backend: The backend package whose base the class must subclass.

        Test scenario:
            The `TYPE_CHECKING` alias promises mypy that `self` inside a mixin is the backend base. That is
            only true because the composed class supplies it — assert it does, in every backend, so the
            promise is not vacuous.
        """
        base_module, base_name = BACKENDS[backend]
        composed = getattr(_import_or_skip(module), name)
        base = getattr(_import_or_skip(base_module), base_name)
        assert issubclass(composed, base), f"{name} must subclass {base_name}"


@functools.cache
def _self_builders() -> tuple:
    """Return `(module, FunctionDef)` for every `-> Self` builder under `src/digitalearth/`.

    A `-> Self` **classmethod** is an alternate constructor — `Scene3D.from_figure(figure)` builds a scene and
    returns it — so it is not a builder and is left out: `Self` is the right annotation there (it is the
    subclass that comes back), but returning `self` is not.
    """
    found = []
    for path in sorted((ROOT / "src" / "digitalearth").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_function and node.returns is not None:
                constructor = any(
                    isinstance(decorator, ast.Name) and decorator.id == "classmethod"
                    for decorator in node.decorator_list
                )
                if ast.unparse(node.returns) == "Self" and not constructor:
                    module = path.relative_to(ROOT / "src").with_suffix("").as_posix()
                    found.append((module.replace("/", "."), node))
    return tuple(found)


def _builder_ids() -> list:
    """Return `module:function` ids for parametrising over every `-> Self` builder."""
    return [f"{module}:{node.name}" for module, node in _self_builders()]


def _own_returns(function) -> list:
    """Return the `Return` nodes in a function's own scope, skipping nested functions and classes."""
    returns, stack = [], list(function.body)
    while stack:
        node = stack.pop()
        nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        if isinstance(node, nested):
            continue
        if isinstance(node, ast.Return):
            returns.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return returns


class TestSelfReturningBuilders:
    """Tests for the PEP 673 `Self` return type the fluent builders carry."""

    def test_the_scan_finds_the_builders(self):
        """The source scan finds at least the 67 `-> Self` builders #172 annotated.

        Test scenario:
            The per-builder assertions below are parametrised over this scan, so an empty one would report
            as a clean pass with nothing checked.
        """
        found = _self_builders()
        assert len(found) >= BUILDER_FLOOR, (
            f"expected >={BUILDER_FLOOR} `-> Self` builders, found {len(found)}: {_builder_ids()}"
        )

    @pytest.mark.parametrize("index", range(len(_self_builders())), ids=_builder_ids())
    def test_builder_returns_self_or_delegates_to_one_that_does(self, index):
        """Every `-> Self` function returns `self`, or the result of another `-> Self` method on `self`.

        Args:
            index: Position of the builder in the source scan.

        Test scenario:
            `-> Self` is the promise that makes `m.field(dem).basemap()` type-check against the composed
            map rather than the mixin. mypy cannot police it across these modules — `no-any-return` is
            baselined off in `web.vector`, `web.threed`, `web.base`, `interactive.vector` and others, so a
            builder that returned an untyped engine object would type-check clean and break chaining at run
            time. Returns in nested helpers (`def apply(widget)`, DynamicMap callbacks) are ignored: they
            belong to the closure, not the builder.
        """
        module, node = _self_builders()[index]
        chainable = {other.name for _, other in _self_builders()}
        returns = _own_returns(node)
        assert returns, f"{module}.{node.name} is annotated `-> Self` but never returns"
        for statement in returns:
            value = statement.value
            returns_self = isinstance(value, ast.Name) and value.id == "self"
            delegates = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "self"
                and value.func.attr in chainable
            )
            assert returns_self or delegates, (
                f"{module}.{node.name} is annotated `-> Self` but returns "
                f"{ast.unparse(value) if value else 'bare return'}; it must return `self` or delegate to "
                "another `-> Self` method"
            )

    @pytest.mark.parametrize(
        "backend", sorted(BACKENDS), ids=lambda b: b.split(".")[-1]
    )
    def test_no_builder_still_quotes_a_class_name(self, backend):
        """No backend method annotates its return as a quoted mixin or base-class name.

        Args:
            backend: The backend package to scan.

        Test scenario:
            The annotation these builders carried before #172 — `-> "VectorMixin"` — named the mixin rather
            than the composed class, so a chained call type-checked as the mixin and lost every method the
            other mixins contribute. A new builder copy-pasted from an old one would reintroduce it silently;
            nothing but this catches that, because the wrong annotation still runs perfectly.
        """
        offenders = []
        for path in sorted(_module_path(backend).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                if not (is_function and isinstance(node.returns, ast.Constant)):
                    continue
                annotation = node.returns.value
                quoted = isinstance(annotation, str) and (
                    annotation.endswith("Mixin")
                    or annotation.endswith("Base")
                    or annotation in {"Map", "InteractiveMap", "WebMap", "Scene3D"}
                )
                if quoted:
                    offenders.append(f"{path.name}:{node.name} -> {annotation!r}")
        assert not offenders, (
            f"{backend} builders must be annotated `-> Self`, not a quoted class name: {offenders}"
        )


#: How a layer reaches a figure: the funnels a builder calls on ``self`` to register one. Named **exactly**
#: rather than by a ``_describe`` prefix, which is what round 2's N1 flagged: a prefix match turns any future
#: ``self._described_opts()`` *reader* into a registrar. Measured off the tree — ``self._describe_layer``
#: (static, and ``_draw`` which wraps it), ``self._describe_graticule`` and ``self._describe_basemap`` (the
#: two that *replace* a layer rather than add one), ``self._add_described_layer`` (the 3-D tier's funnel) and
#: ``add_layer``/``add_underlay`` (interactive and web).
REGISTRARS = frozenset(
    {
        "_draw",
        "_describe_layer",
        "_describe_graticule",
        "_describe_basemap",
        "_add_described_layer",
        "add_layer",
        "add_underlay",
    }
)

#: The base and composed classes the contract is policed over beside the mixins, as
#: `module -> class names`. The mixins alone are not the chaining surface: `Scene` owns `add_layer`,
#: `colorbar`, `legend`, `set_title`, `move_layer`, `replace_layer` and `set_visible`, and `GeoLayerBase`,
#: `WebMapBase`, `Scene3DBase` and `WebMapBase`'s composed `WebMap` own more. Round 2's M7 measured the
#: previous scan at 3 methods on `*Mixin` classes only, which left every one of these unopened.
COMPOSED_CLASSES = {
    "digitalearth.static.scene": ("Scene",),
    "digitalearth.static.map": ("Map",),
    "digitalearth.static.textured_globe": ("TexturedGlobe",),
    "digitalearth.static.maps.base": ("GeoLayerBase",),
    "digitalearth.interactive.base": ("InteractiveMapBase",),
    "digitalearth.interactive.map": ("InteractiveMap",),
    "digitalearth.three_d.base": ("Scene3DBase",),
    "digitalearth.three_d.scene3d": ("Scene3D",),
    "digitalearth.web.base": ("WebMapBase",),
    "digitalearth.web.map": ("WebMap",),
}

#: How many classes and public methods the scan reached when this guard was widened. Floors, not
#: equalities — a new mixin or a new method is normal — so that a scan broken into reaching nothing cannot
#: report as a clean pass. The previous scan's numbers, for contrast: 23 classes and the 3 public `-> None`
#: methods M7 measured.
CONTRACT_CLASS_FLOOR = 40
CONTRACT_METHOD_FLOOR = 265

#: The 3-D tier's own return convention, shared by nine builders.
_THREE_D_ACTOR = (
    "the 3-D tier's builders hand back the PyVista actor they made rather than the scene (round 1's M13); "
    "converting the tier is a `src/` change, not this guard's"
)

#: The public methods that register a layer, or delegate to a builder, and deliberately do **not** answer
#: `Self`, as `module:Class.method -> why`. Everything else the two rules below flag is a defect.
#:
#: This table is the explicit opt-out the contract needs for a terminal method, and it is where the three
#: shapes round 2's N1 named belong — a teardown that re-registers to restore layer order, and an event
#: callback a caller must not chain on, are both flagged by rule 1 and must be entered here with a reason
#: rather than escaping the scan silently. (N1's third shape, a description *reader*, is no longer flagged
#: at all: see :data:`REGISTRARS`.)
CONTRACT_CARVE_OUTS = {
    "digitalearth.static.maps.inset:InsetMixin.mark_extent": (
        "hands back the rectangle it drew on the locator map, which is what the method is for"
    ),
    "digitalearth.static.maps.decoration:DecorationMixin.stock_img": (
        "hands back the backdrop artist a caller restyles; pinned by "
        "tests/static/test_decoration_chaining.py"
    ),
    "digitalearth.three_d.base:Scene3DBase.add_mesh": _THREE_D_ACTOR,
    "digitalearth.three_d.base:Scene3DBase.add_volume": _THREE_D_ACTOR,
    "digitalearth.three_d.globe:GlobeMixin.globe": _THREE_D_ACTOR,
    "digitalearth.three_d.point_cloud:PointCloudMixin.point_cloud": _THREE_D_ACTOR,
    "digitalearth.three_d.terrain:TerrainMixin.terrain": _THREE_D_ACTOR,
    "digitalearth.three_d.vector:VectorMixin.extruded_polygons": _THREE_D_ACTOR,
    "digitalearth.three_d.vector:VectorMixin.vectors": _THREE_D_ACTOR,
    "digitalearth.three_d.volume:VolumeMixin.isosurface": _THREE_D_ACTOR,
    "digitalearth.three_d.volume:VolumeMixin.volume": _THREE_D_ACTOR,
}


def _self_calls(function, nested: bool = False) -> set:
    """Return the `self.<name>(...)` methods called in a function.

    Args:
        function: The `FunctionDef` to read.
        nested: Whether to read into nested functions and lambdas as well. `False` (the default) keeps a
            call inside a closure with the closure, which is what the delegation rule wants. `True` is what
            the registration rule wants: a layer registered from inside a `def apply(widget)` is still
            registered, and the closure was round 2's M7 escape route I.

    Returns:
        The attribute names called on `self`. A nested **class** is skipped either way — its methods are
        another class's `self`.
    """
    names, stack = set(), list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.ClassDef):
            continue
        shallow = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        if isinstance(node, shallow) and not nested:
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            names.add(node.func.attr)
        stack.extend(ast.iter_child_nodes(node))
    return names


def _class_members(klass: ast.ClassDef) -> dict:
    """Return `method name -> FunctionDef` for one class body.

    Args:
        klass: The `ClassDef` to read.

    Returns:
        Its own functions, so the registration walk can follow a private helper by name.
    """
    return {
        node.name: node
        for node in klass.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _registrar_reach(members: dict, name: str, seen=None) -> set:
    """Return the registration funnels one method reaches, through its class's own private helpers.

    Args:
        members: The class's `method name -> FunctionDef` map, as :func:`_class_members` builds it.
        name: The method to walk from.
        seen: Method names already walked, so a cycle terminates.

    Returns:
        The names from :data:`REGISTRARS` the method reaches. The walk is transitive through **private**
        helpers on the same class, which is round 2's M7 escape route C: `mark_extent` calls `self._mark`,
        which calls `self.add_layer`, and a one-level scan sees neither.
    """
    seen = set() if seen is None else seen
    if name in seen:
        return set()
    seen.add(name)
    node = members.get(name)
    if node is None:
        return set()
    found = set()
    for called in _self_calls(node, nested=True):
        if called in REGISTRARS:
            found.add(called)
        elif called.startswith("_"):
            found |= _registrar_reach(members, called, seen)
    return found


@functools.cache
def _contract_classes() -> tuple:
    """Return `(module, ClassDef)` for every class the builder contract is policed over.

    Every `*Mixin` in the four backend packages, plus the base and composed classes
    :data:`COMPOSED_CLASSES` names. Renderers, glyph internals and the private helper classes beside them
    are deliberately out of scope: they are machinery a caller never chains on, and matching a builder name
    against them is what makes `Renderer.draw_layer` look like a method that should answer `Self`.
    """
    found = []
    for backend in BACKENDS:
        for path in sorted(_module_path(backend).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for klass in tree.body:
                if isinstance(klass, ast.ClassDef) and klass.name.endswith("Mixin"):
                    found.append((f"{backend}.{path.stem}", klass))
    for module, names in COMPOSED_CLASSES.items():
        path = _module_path(module).with_suffix(".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for klass in tree.body:
            if isinstance(klass, ast.ClassDef) and klass.name in names:
                found.append((module, klass))
    return tuple(found)


@functools.cache
def _contract_methods() -> tuple:
    """Return `(id, members, FunctionDef)` for every public method on a contract class.

    `id` is `module:Class.method`, the key :data:`CONTRACT_CARVE_OUTS` uses, and `members` is the owning
    class's method map so a rule can walk it.
    """
    found = []
    for module, klass in _contract_classes():
        members = _class_members(klass)
        for name, node in sorted(members.items()):
            if not name.startswith("_"):
                found.append((f"{module}:{klass.name}.{name}", members, node))
    return tuple(found)


def _answers_self(node) -> bool:
    """Return whether a function is annotated `-> Self`."""
    return node.returns is not None and ast.unparse(node.returns) == "Self"


class TestTheContractScanReachesTheComposition:
    """Tests that the two rules below are read over the whole chaining surface, not a corner of it.

    Round 2's M7 measured the previous scan at **3** public methods: it filtered classes on
    `name.endswith("Mixin")` and opened only the four backend directories, so `Scene`, `GeoLayerBase`,
    `Map`, `WebMapBase`, `WebMap`, `Scene3DBase`, `Scene3D`, `InteractiveMapBase` and `InteractiveMap`
    were never read at all — and `Scene` is where `add_layer` itself lives. These assertions pin the
    widened scope, because a rule is worth exactly what its scan reaches.
    """

    def test_the_scan_opens_every_base_and_composed_class(self):
        """All ten classes :data:`COMPOSED_CLASSES` names are in scope.

        Test scenario:
            The scope defect M7 named, asserted directly: a filter that reads `*Mixin` classes only
            satisfies every rule below over a tree where `Scene.add_layer` has regressed.
        """
        opened = {f"{module}:{klass.name}" for module, klass in _contract_classes()}
        expected = {
            f"{module}:{name}"
            for module, names in COMPOSED_CLASSES.items()
            for name in names
        }
        assert expected <= opened, (
            f"these classes must be in the contract scan: {sorted(expected - opened)}"
        )

    def test_the_scan_still_opens_every_mixin(self):
        """Every mixin the declaration scan finds is a contract class too.

        Test scenario:
            The widened scope must be a superset of the old one, so a mixin cannot drop out of the rules
            while the composed classes are added.
        """
        mixins = {f"{module}:{name}" for _, module, name, _, _ in _mixins()}
        opened = {f"{module}:{klass.name}" for module, klass in _contract_classes()}
        assert mixins <= opened, (
            f"these mixins fell out of the contract scan: {sorted(mixins - opened)}"
        )

    def test_the_scan_reaches_the_methods_it_was_measured_over(self):
        """The scan reaches at least the classes it reached when it was widened.

        Test scenario:
            Every assertion in the next class iterates this scan and passes over an empty one, which is
            how a scan that reached 3 methods read as a guard over 220.
        """
        assert len(_contract_classes()) >= CONTRACT_CLASS_FLOOR, (
            f"expected >={CONTRACT_CLASS_FLOOR} contract classes, found {len(_contract_classes())}"
        )

    def test_the_scan_reaches_the_public_methods_it_was_measured_over(self):
        """The scan reaches at least the public-method count it was measured at.

        Test scenario:
            The companion floor to the class count: a filter that opened every class and then read no
            method off it would satisfy the one above.
        """
        assert len(_contract_methods()) >= CONTRACT_METHOD_FLOOR, (
            f"expected >={CONTRACT_METHOD_FLOOR} public contract methods, found {len(_contract_methods())}"
        )


class TestNoBuilderQuietlyDeclinesToChain:
    """Tests that a method which *builds* answers `Self`, so the call chains.

    :class:`TestSelfReturningBuilders` checks that a method annotated `-> Self` really returns `self`. A
    builder annotated anything else satisfies it by having nothing to check, which is how
    `static.maps.projection.graticule`, `set_domain` and `set_global` sat beside a `set_bounds` that
    returned `Self` for a whole release: `graticule` registered `graticule-1` and drew, and handed back
    nothing. These two rules read the *other* direction — a method that does a builder's work must not
    declare a builder's opposite.

    The test is **"not `Self`"**, not "`-> None`". Round 2's M7 showed that an `-> None` test leaves four
    other ways to write the same defect: no annotation at all, the string `'None'`, handing back the artist
    (the pre-conversion convention), and `-> Any`. All four now read as the same finding.
    """

    def test_no_public_method_registers_a_layer_without_answering_self(self):
        """A method that puts a layer in the figure hands the map back, so the call chains.

        Test scenario:
            Registering is what makes a method a builder: the layer it describes is addressable by id,
            carries a `visible` flag and is redrawn from a stored figure, and every other such method on
            the 2-D tiers answers `Self`. One that answers anything else breaks the chain at exactly the
            point a caller cannot see coming. The known exceptions are enumerated in
            :data:`CONTRACT_CARVE_OUTS` with a reason each, so a new one has to be argued for rather than
            merely written.
        """
        offenders = [
            f"{key} -> {'<none>' if node.returns is None else ast.unparse(node.returns)}"
            f" registers via {sorted(_registrar_reach(members, key.rsplit('.', 1)[1]))}"
            for key, members, node in _contract_methods()
            if not _answers_self(node)
            and key not in CONTRACT_CARVE_OUTS
            and _registrar_reach(members, key.rsplit(".", 1)[1])
        ]
        assert offenders == [], (
            "these register a layer and must be annotated `-> Self`, returning self (or be entered in "
            f"CONTRACT_CARVE_OUTS with a reason): {offenders}"
        )

    def test_no_public_method_delegates_to_a_builder_without_answering_self(self):
        """A method whose work is another builder's answers what that builder answers.

        Test scenario:
            `set_domain` and `set_global` are `set_bounds` with the argument worked out — one resolves a
            named region, the other reads the projection's own domain — so throwing its `Self` away was
            the whole defect. A terminal action is not caught by this: `render` calls `_apply_frame`,
            which is not a builder, and `show` calls `super().show()`, which is not a call on `self`.
        """
        chainable = {
            node.name
            for _, klass in _contract_classes()
            for node in _class_members(klass).values()
            if _answers_self(node)
        }
        offenders = [
            f"{key} -> {'<none>' if node.returns is None else ast.unparse(node.returns)}"
            f" delegates to {sorted(_self_calls(node) & chainable)}"
            for key, _, node in _contract_methods()
            if not _answers_self(node)
            and key not in CONTRACT_CARVE_OUTS
            and _self_calls(node) & chainable
        ]
        assert offenders == [], (
            "these delegate to a `-> Self` builder and must answer what it answers (or be entered in "
            f"CONTRACT_CARVE_OUTS with a reason): {offenders}"
        )

    def test_every_carve_out_names_a_method_the_scan_finds(self):
        """Each entry in the carve-out table is a method that exists and is in scope.

        Test scenario:
            A table of names is a second place for the tree to drift away from. An entry whose method was
            renamed, made private or moved would silently stop exempting anything — and, worse, would read
            as evidence that the rule had been thought about where it no longer applies.
        """
        reachable = {key for key, _, _ in _contract_methods()}
        stale = sorted(set(CONTRACT_CARVE_OUTS) - reachable)
        assert stale == [], (
            f"these carve-outs name no public method the scan reaches: {stale}"
        )


#: One crafted class per way round 2's M7 proved a registrar could evade the rule, as
#: `route -> (source, class, method)`. Fed to the helpers the rule itself uses rather than to a
#: reimplementation of them, so narrowing the rule again fails these rather than passing quietly.
ESCAPE_ROUTES = {
    "A-plain-none": (
        """
        class Offender:
            def spaghetti(self, data) -> None:
                self._draw(data)
        """,
        "Offender",
        "spaghetti",
    ),
    "C-private-hop": (
        """
        class Offender:
            def mark_extent(self, extent) -> None:
                self._mark(extent)

            def _mark(self, extent):
                return self.add_layer(extent)
        """,
        "Offender",
        "mark_extent",
    ),
    "D-no-annotation": (
        """
        class Offender:
            def field(self, data):
                return self._draw(data)
        """,
        "Offender",
        "field",
    ),
    "E-string-none": (
        """
        class Offender:
            def field(self, data) -> 'None':
                self._draw(data)
        """,
        "Offender",
        "field",
    ),
    "F-hands-back-the-artist": (
        """
        class Offender:
            def field(self, data) -> Any:
                return self._draw(data)
        """,
        "Offender",
        "field",
    ),
    "H-three-d-funnel": (
        """
        class Offender:
            def terrain(self, data) -> Any:
                return self._add_described_layer(data)
        """,
        "Offender",
        "terrain",
    ),
    "I-inside-a-closure": (
        """
        class Offender:
            def dynamic(self, data) -> None:
                def apply(widget):
                    return self._draw(data)

                apply(None)
        """,
        "Offender",
        "dynamic",
    ),
}

#: Crafted classes the rule must leave alone, as `shape -> (source, class, method)`. The first is round 2's
#: N1: a terminal that only *reads* a description through a name beginning `_describe`, which the previous
#: prefix match turned into a registrar. The second is the private helper the previous rule excluded by
#: reading public methods only, kept here so that exclusion is pinned rather than assumed.
LEGITIMATE_SHAPES = {
    "a-description-reader": (
        """
        class Legitimate:
            def describe(self) -> None:
                print(self._described_opts())
        """,
        "Legitimate",
        "describe",
    ),
    "a-terminal-that-reads-the-axes": (
        """
        class Legitimate:
            def render(self) -> None:
                self._apply_frame()
                self._settle()
        """,
        "Legitimate",
        "render",
    ),
}


def _crafted(source: str, klass: str) -> dict:
    """Return the method map of one crafted class, so a rule can be run against it.

    Args:
        source: An indented class definition, as the tables above hold it.
        klass: The class to read out of it.

    Returns:
        Its `method name -> FunctionDef` map, as :func:`_class_members` builds it for a real class.
    """
    tree = ast.parse(textwrap.dedent(source))
    found = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == klass
    )
    return _class_members(found)


def _registration_rule(source: str, klass: str, method: str) -> bool:
    """Return whether the registration rule flags one crafted method.

    Args:
        source: The crafted class definition.
        klass: The class in it.
        method: The method the rule is read over.

    Returns:
        `True` when the method both fails to answer `Self` and reaches a registrar — the exact condition
        `test_no_public_method_registers_a_layer_without_answering_self` applies to the tree.
    """
    members = _crafted(source, klass)
    node = members[method]
    return not _answers_self(node) and bool(_registrar_reach(members, method))


class TestTheRegistrationRuleHasTeethOnEveryEscapeRoute:
    """Tests the rule against one planted offender per escape route round 2's M7 measured.

    M7's finding was not that the rule was wrong about the two shapes it was written for, but that it
    covered only those two: a private hop, a missing annotation, a string annotation, an artist return,
    the 3-D tier's funnel and a closure all walked past it. Each is planted here, as source, and run
    through the same helpers the tree is read with.
    """

    @pytest.mark.parametrize("route", sorted(ESCAPE_ROUTES))
    def test_the_rule_flags_the_planted_registrar(self, route):
        """Each planted registrar is flagged.

        Args:
            route: The entry in :data:`ESCAPE_ROUTES` under test.

        Test scenario:
            A ratchet is worth what it catches, and the only way to know is to plant the thing it is
            supposed to catch. These seven were measured as passing undetected before this guard was
            widened.
        """
        source, klass, method = ESCAPE_ROUTES[route]
        assert _registration_rule(source, klass, method), (
            f"escape route {route} is not flagged; the rule has no teeth on it"
        )

    @pytest.mark.parametrize("shape", sorted(LEGITIMATE_SHAPES))
    def test_the_rule_leaves_the_legitimate_shape_alone(self, shape):
        """Each shape that is not a builder is left alone.

        Args:
            shape: The entry in :data:`LEGITIMATE_SHAPES` under test.

        Test scenario:
            Round 2's N1: the rule's `_describe` **prefix** match made any future
            `self._described_opts()` reader a registrar. The funnels are named exactly now, and this is
            what says so.
        """
        source, klass, method = LEGITIMATE_SHAPES[shape]
        assert not _registration_rule(source, klass, method), (
            f"{shape} is a false positive; the rule should not flag it"
        )

    def test_a_planted_builder_that_answers_self_is_not_flagged(self):
        """A registrar annotated `-> Self` is what the rule asks for, so it is not flagged.

        Test scenario:
            The other half of the rule's condition. Without this the two assertions above would hold for
            a rule that flagged every registrar, `-> Self` ones included — which would make the contract
            impossible to satisfy rather than enforced.
        """
        source = """
        class Builder:
            def field(self, data) -> Self:
                self._draw(data)
                return self
        """
        assert not _registration_rule(source, "Builder", "field"), (
            "a `-> Self` registrar satisfies the contract and must not be flagged"
        )


def _delegation_rule(source: str, klass: str, method: str) -> bool:
    """Return whether the delegation rule flags one crafted method.

    Args:
        source: The crafted class definition.
        klass: The class in it.
        method: The method the rule is read over.

    Returns:
        `True` when the method fails to answer `Self` and calls a `-> Self` builder on `self`. The
        chainable set is the tree's own, so this also pins that `set_bounds` is still a builder.
    """
    members = _crafted(source, klass)
    node = members[method]
    chainable = {
        other.name
        for _, found in _contract_classes()
        for other in _class_members(found).values()
        if _answers_self(other)
    }
    return not _answers_self(node) and bool(_self_calls(node) & chainable)


class TestTheDelegationRuleReadsEveryAnnotation:
    """Tests the delegation rule against the annotations the `-> None` test walked past.

    The rule had teeth on `-> None` — that is how `set_domain` and `set_global` were found — and on
    nothing else. A method that threw a builder's `Self` away under any other annotation was invisible,
    which is the same hole round 2's M7 measured on the registration rule.
    """

    def test_a_none_annotated_delegation_is_flagged(self):
        """The shape the rule was written for is still flagged.

        Test scenario:
            The control. Without it the two assertions below would hold for a rule that had stopped
            reading `-> None` while gaining the other annotations.
        """
        source = """
        class Offender:
            def set_global(self) -> None:
                self.set_bounds(-180.0, -90.0, 180.0, 90.0)
        """
        assert _delegation_rule(source, "Offender", "set_global"), (
            "a `-> None` method whose work is `set_bounds` must be flagged"
        )

    def test_an_any_annotated_delegation_is_flagged(self):
        """`-> Any` over a builder is the same defect, and now reads as one.

        Test scenario:
            `-> Any` is the annotation the pre-conversion builders carried, so it is the one a
            copy-paste reintroduces. The previous rule read `-> None` exactly and let it through.
        """
        source = """
        class Offender:
            def set_global(self) -> Any:
                return self.set_bounds(-180.0, -90.0, 180.0, 90.0)
        """
        assert _delegation_rule(source, "Offender", "set_global"), (
            "an `-> Any` method whose work is `set_bounds` must be flagged"
        )

    def test_a_terminal_action_is_not_flagged(self):
        """A terminal that calls private helpers only is left alone.

        Test scenario:
            `render`'s shape: it calls `self._apply_frame()`, which is not a builder, so the rule must
            not ask it to chain. A rule that flagged every `-> None` method would make the contract
            impossible to satisfy instead of enforcing it.
        """
        source = """
        class Terminal:
            def render(self) -> None:
                self._apply_frame()
        """
        assert not _delegation_rule(source, "Terminal", "render"), (
            "a terminal action calls no builder and must not be flagged"
        )


class TestBuilderChaining:
    """Tests that the engine-free registration builders chain on the composed class."""

    @pytest.mark.parametrize(
        "module, name, builder",
        [
            ("digitalearth.web.map", "WebMap", "add_layer"),
            ("digitalearth.web.map", "WebMap", "add_underlay"),
            ("digitalearth.interactive.map", "InteractiveMap", "add_layer"),
        ],
        ids=["web-add_layer", "web-add_underlay", "interactive-add_layer"],
    )
    def test_registration_returns_the_composed_map(self, module, name, builder):
        """The low-level registration builders return the composed map itself, so calls chain.

        Args:
            module: Module holding the composed class.
            name: The composed class.
            builder: The registration method under test.

        Test scenario:
            These three are the terminus every other `-> Self` builder delegates to, and the only ones that
            run without an optional engine — the existing `test_add_layer_chains` / `test_add_layer_chains_twice`
            sit behind `importorskip("maplibre")` / `importorskip("geoviews")`, so neither runs in the `dev`
            environment CI gates on, and `add_underlay` has no chaining test at all. Assert both identity and
            exact type: `Self` promises the *composed* class, not the mixin that defines the method.
        """
        composed = getattr(_import_or_skip(module), name)
        instance = composed()
        result = getattr(instance, builder)("layer-a")
        assert result is instance, (
            f"{name}.{builder}() must return the same map instance"
        )
        assert type(result) is composed, (
            f"{name}.{builder}() must return the composed {name}, got {type(result).__name__}"
        )

    def test_chained_calls_register_in_order(self):
        """A chained `add_layer` / `add_underlay` sequence stacks underlays below layers.

        Test scenario:
            Chaining is only useful if each call in the chain still lands its side effect. Append two layers
            and slide one underneath, and the registry must read `['tiles', 'a', 'b']`.
        """
        web_map = _import_or_skip("digitalearth.web.map").WebMap()
        result = web_map.add_layer("a").add_layer("b").add_underlay("tiles")
        assert result is web_map, "the whole chain must return the same map"
        assert web_map.layers == ["tiles", "a", "b"], (
            f"expected underlay first then both layers, got {web_map.layers}"
        )
