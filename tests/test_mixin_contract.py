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
``static.map.Map`` lists ``GeoLayerBase`` last, so C3 would still linearise to the very same nine class **names**
that ``tests/static/test_map_composition.py`` pins — while every mixin quietly gained a real base class, and with it a
different ``super()`` chain and a different ``__init__``. Nothing else in the suite compares ``__bases__``, so
this file does.

The second guard is on the PEP 673 ``Self`` return type the 67 fluent builders now carry in place of the mixin
name they used to quote. ``-> Self`` is a promise that the call returns the *composed* map, so builders chain;
mypy cannot enforce it everywhere because ``no-any-return`` is baselined off in several of these modules, so the
promise is checked here instead: every ``-> Self`` method must return ``self``, or delegate to another ``-> Self``
method that does.

Both scans read the source with :mod:`ast` rather than importing, so all 27 mixins are covered in every
environment — including the five ``three_d`` modules that need PyVista, which the ``dev`` env does not have.
The runtime assertions are the belt to that braces, and skip where an optional engine is missing.
"""

import ast
import functools
import importlib
import pathlib

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


@functools.lru_cache(maxsize=None)
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
    and friends), listing the base *first* puts it both before and after its own subclasses, and C3 cannot
    linearize it — mypy then falls back to ``[cls, object]`` and every attribute on the class silently
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


@functools.lru_cache(maxsize=None)
def _self_builders() -> tuple:
    """Return `(module, FunctionDef)` for every `-> Self` function under `src/digitalearth/`."""
    found = []
    for path in sorted((ROOT / "src" / "digitalearth").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_function and node.returns is not None:
                if ast.unparse(node.returns) == "Self":
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
            `-> Self` is the promise that makes `m.add_raster(dem).basemap()` type-check against the composed
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


class TestBuilderChaining:
    """Tests that the engine-free registration builders chain on the composed class."""

    @pytest.mark.parametrize(
        "module, name, builder",
        [
            ("digitalearth.web.map", "WebMap", "add_layer"),
            ("digitalearth.web.map", "WebMap", "add_underlay"),
            ("digitalearth.interactive.map", "InteractiveMap", "add_element"),
        ],
        ids=["web-add_layer", "web-add_underlay", "interactive-add_element"],
    )
    def test_registration_returns_the_composed_map(self, module, name, builder):
        """The low-level registration builders return the composed map itself, so calls chain.

        Args:
            module: Module holding the composed class.
            name: The composed class.
            builder: The registration method under test.

        Test scenario:
            These three are the terminus every other `-> Self` builder delegates to, and the only ones that
            run without an optional engine — the existing `test_add_layer_chains` / `test_add_element_chains`
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
