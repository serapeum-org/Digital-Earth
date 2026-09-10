"""Keyed-XYZ basemap presets (#160) — the provider definitions and the static-tier dispatch.

The web and interactive dispatch live in ``test_web_basemaps.py`` /
``test_interactive_basemaps.py``, which is what the per-tier CI tasks glob.

The definitions in :mod:`digitalearth.base.basemaps` are pure data, so most of this runs in the lean ``dev``
env. The tier-dispatch tests ``importorskip`` their engine. Nothing here needs a real Planet key or reaches
the network: a fake key is set in the environment, and the static tier's tile fetch is replaced by a spy.
"""

import logging

import pytest

from digitalearth.base.basemaps import (
    KEYED_BASEMAPS,
    KeyedTileSource,
    get_keyed_basemap,
    is_keyed_basemap,
    planet_nicfi,
)

FAKE_KEY = "FAKE-KEY-NOT-REAL"
#: The Amazon — inside the NICFI band. (west, south, east, north)
TROPICAL = (-60.0, -5.0, -55.0, 0.0)
#: The Netherlands — outside it.
TEMPERATE = (5.0, 52.0, 6.0, 53.0)


class TestPlanetNicfi:
    """The NICFI preset: mosaic-id resolution, coverage, attribution."""

    @pytest.mark.parametrize(
        "flavour, expected",
        [
            ("analytic", "planet_medres_normalized_analytic_2024-01_mosaic"),
            ("visual", "planet_medres_visual_2024-01_mosaic"),
        ],
    )
    def test_the_date_and_flavour_resolve_the_mosaic_id(self, flavour, expected):
        """Each flavour is a distinct Planet product with its own id stem.

        Args:
            flavour: The NICFI product kind.
            expected: The mosaic id it must resolve to.

        Test scenario:
            The mosaic id embeds the month, so "which mosaic for this date" is provider logic rather than
            something the caller should have to spell out.
        """
        assert planet_nicfi("2024-01", flavour=flavour).params["mosaic"] == expected

    def test_an_explicit_mosaic_overrides_the_derived_one(self):
        """Planet's naming has variants, so the derived id must be overridable.

        Test scenario:
            The biannual pre-2020 mosaics are named differently; without an override they would be
            unreachable through this preset.
        """
        source = planet_nicfi(
            "2024-01", mosaic="planet_medres_normalized_analytic_2019-06_2019-11_mosaic"
        )
        assert source.params["mosaic"].endswith("2019-06_2019-11_mosaic")

    @pytest.mark.parametrize(
        "date", ["2024-1", "24-01", "Jan 2024", "2024-13", "2024-00", ""]
    )
    def test_a_malformed_month_is_refused(self, date):
        """A bad month must fail here, not as an opaque 404 from Planet.

        Args:
            date: The rejected month string.

        Test scenario:
            Requesting a mosaic that cannot exist otherwise surfaces as cleopatra's ConnectionError after
            its retries, which says nothing about the real mistake.
        """
        with pytest.raises(ValueError, match="YYYY-MM"):
            planet_nicfi(date)

    def test_an_unknown_flavour_names_the_two_that_exist(self):
        """The error has to say what the alternatives are, and what they mean.

        Test scenario:
            'visual' vs 'analytic' is a product choice, not a styling flag, so guessing is unhelpful.
        """
        with pytest.raises(ValueError, match="analytic"):
            planet_nicfi("2024-01", flavour="rgb")

    def test_the_attribution_and_licence_are_carried(self):
        """NICFI is free for non-commercial use only, so the terms travel with the source.

        Test scenario:
            The attribution is what a rendered map displays; the licence note is why it matters.
        """
        source = planet_nicfi("2024-01")
        assert "Planet Labs" in source.attribution, (
            f"missing attribution: {source.attribution!r}"
        )
        assert "non-commercial" in source.attribution, (
            f"missing licence note: {source.attribution!r}"
        )

    def test_the_coverage_is_the_tropics(self):
        """NICFI publishes roughly 30°N–30°S; the bounds are what the extent guard reads."""
        assert planet_nicfi("2024-01").bounds == (-180.0, -30.0, 180.0, 30.0)


class TestCredential:
    """The key is read from the environment, never hard-coded, and never echoed."""

    def test_an_explicit_key_wins_over_the_environment(self, monkeypatch):
        """A caller-supplied key takes precedence, so a script can override the ambient one.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        monkeypatch.setenv("PLANET_API_KEY", "FROM-ENV")
        assert "EXPLICIT" in planet_nicfi("2024-01").tile_url(api_key="EXPLICIT")

    def test_the_environment_supplies_the_key_when_none_is_passed(self, monkeypatch):
        """The documented path: set PLANET_API_KEY and call basemap() with no secret in the code.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)
        assert FAKE_KEY in planet_nicfi("2024-01").tile_url()

    @pytest.mark.parametrize("value", ["", None])
    def test_a_missing_key_names_the_variable_to_set(self, monkeypatch, value):
        """An unset *or empty* variable must produce an actionable error, not a blank map.

        Args:
            monkeypatch: pytest's environment patcher.
            value: The unhelpful environment state — unset (``None``) or empty.

        Test scenario:
            An empty string is the likelier mistake (an unset shell variable expands to one), and it would
            otherwise build a URL with `api_key=` and 401 at the service.
        """
        if value is None:
            monkeypatch.delenv("PLANET_API_KEY", raising=False)
        else:
            monkeypatch.setenv("PLANET_API_KEY", value)
        with pytest.raises(ValueError, match="PLANET_API_KEY"):
            planet_nicfi("2024-01").tile_url()

    def test_the_key_is_not_in_the_repr_of_the_source(self, monkeypatch):
        """The source is a dataclass; its repr must not carry a credential.

        Args:
            monkeypatch: pytest's environment patcher.

        Test scenario:
            The key is resolved only when a URL is built, so the object itself stays safe to log or print.
        """
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)
        assert FAKE_KEY not in repr(planet_nicfi("2024-01"))


class TestTileUrl:
    """URL templating: placeholders substituted, tile coordinates preserved."""

    def test_the_tile_coordinates_survive_for_the_engine(self, monkeypatch):
        """``{z}``/``{x}``/``{y}`` must remain — the rendering engine substitutes those, not us.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)
        url = planet_nicfi("2024-01").tile_url()
        assert "{z}/{x}/{y}" in url, f"tile placeholders were consumed: {url}"

    def test_no_placeholder_is_left_unfilled(self, monkeypatch):
        """Everything except the tile coordinates must be resolved.

        Args:
            monkeypatch: pytest's environment patcher.

        Test scenario:
            A leftover ``{mosaic}`` or ``{api_key}`` would be requested literally and 404.
        """
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)
        url = planet_nicfi("2024-01").tile_url()
        for leftover in ("{mosaic}", "{api_key}"):
            assert leftover not in url, f"{leftover} was never substituted: {url}"


class TestCheckBounds:
    """The coverage guard — refuse only an extent with no overlap at all."""

    @pytest.mark.parametrize(
        "extent, refused",
        [
            (TROPICAL, False),
            (TEMPERATE, True),
            (
                (-60.0, -40.0, -55.0, -25.0),
                False,
            ),  # straddles the southern edge — partial overlap
            ((-60.0, 40.0, -55.0, 60.0), True),  # entirely north of the band
            (None, False),  # unknown extent — do not guess
        ],
    )
    def test_only_a_disjoint_extent_is_refused(self, extent, refused):
        """A partial overlap still renders tiles where they exist, so only disjoint boxes raise.

        Args:
            extent: The area about to be drawn.
            refused: Whether the guard must raise for it.

        Test scenario:
            Refusing a straddling AOI would block the common case of a map that spans the tropic line.
        """
        source = planet_nicfi("2024-01")
        if refused:
            with pytest.raises(ValueError, match="lies entirely outside"):
                source.check_bounds(extent)
        else:
            source.check_bounds(extent)

    def test_a_source_without_bounds_never_refuses(self):
        """A global service declares no bounds, and must not be guarded against."""
        source = KeyedTileSource(
            name="Example",
            url_template="https://a/{z}/{x}/{y}.png?k={api_key}",
            attribution="Example",
            credential_env="EXAMPLE_KEY",
        )
        source.check_bounds(TEMPERATE)


class TestRegistry:
    """Name resolution, as a ``basemap()`` caller reaches it."""

    @pytest.mark.parametrize("name", ["Planet.NICFI", "planet.nicfi", "PLANET.NICFI"])
    def test_the_preset_name_is_case_insensitive(self, name):
        """Callers type the name; casing must not be a trap.

        Args:
            name: A spelling of the preset name.
        """
        assert is_keyed_basemap(name), f"{name!r} should resolve"
        assert (
            get_keyed_basemap(name, date="2024-01").credential_env == "PLANET_API_KEY"
        )

    @pytest.mark.parametrize(
        "value", ["CartoDark", "https://a/{z}/{x}/{y}.png", None, 5, object()]
    )
    def test_non_presets_are_not_claimed(self, value):
        """A plain provider name, a URL, or an already-built object must pass through untouched.

        Args:
            value: Something that is not a preset name.

        Test scenario:
            Claiming these would break every existing basemap call.
        """
        assert not is_keyed_basemap(value)

    def test_an_unknown_name_lists_what_is_available(self):
        """The error must name the presets that do exist."""
        with pytest.raises(ValueError, match="Planet.NICFI"):
            get_keyed_basemap("Mapbox.Satellite", date="2024-01")

    def test_every_registered_preset_builds_from_its_own_keywords(self):
        """Each registry entry must be callable and return a source carrying a credential variable.

        Test scenario:
            Guards the registry against an entry that is registered but not wired up.
        """
        import inspect

        for name, factory in KEYED_BASEMAPS.items():
            required = [
                p.name
                for p in inspect.signature(factory).parameters.values()
                if p.default is inspect.Parameter.empty
            ]
            # Every preset so far is keyed on a date; a future one without it must not break this test.
            source = get_keyed_basemap(name, **{key: "2024-01" for key in required})
            assert source.credential_env, f"{name} declares no credential variable"
            assert source.attribution, f"{name} declares no attribution"


class TestStaticTierDispatch:
    """``Map.basemap`` resolving a keyed preset (matplotlib + cleopatra)."""

    @pytest.fixture(autouse=True)
    def _spy_on_the_fetch(self, monkeypatch):
        """Replace cleopatra's tile fetch so nothing reaches the network, and record the URLs it wanted.

        Args:
            monkeypatch: pytest's patcher, which restores the real fetch afterwards.
        """
        pytest.importorskip("matplotlib")
        from cleopatra.basemap import tiles as cleo_tiles

        self.requested = []
        self.escaped = []

        def spy(tile, provider, timeout, retries, user_agent="test"):
            url = provider.build_url(x=tile.x, y=tile.y, z=tile.z)
            self.requested.append(url)
            # Emit the record cleopatra emits here, and record whether it survived the suppression.
            logger = logging.getLogger("cleopatra.basemap.tiles")
            handler = logging.Handler()
            handler.emit = self.escaped.append
            logger.addHandler(handler)
            try:
                logger.debug("fetching %s", url)
            finally:
                logger.removeHandler(handler)
            raise ConnectionError("network not used in tests")

        monkeypatch.setattr(cleo_tiles, "fetch_single_tile", spy)
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    @staticmethod
    def _tropical_map():
        """A map over the Amazon with an extent set, which cleopatra requires before tiling."""
        from digitalearth import Map

        m = Map(domain=TROPICAL)
        m.ax.set_xlim(TROPICAL[0], TROPICAL[2])
        m.ax.set_ylim(TROPICAL[1], TROPICAL[3])
        return m

    def test_the_preset_reaches_the_tile_engine_with_its_key(self):
        """The resolved URL — key and mosaic substituted — is what cleopatra fetches.

        Test scenario:
            The whole point of the feature: a caller names the preset and the credential arrives from the
            environment without appearing in their code.
        """
        with pytest.raises(ConnectionError):
            self._tropical_map().basemap(
                "Planet.NICFI", preset={"date": "2024-01", "flavour": "visual"}
            )
        assert self.requested, "no tile fetch was attempted"
        url = self.requested[0]
        assert FAKE_KEY in url, "the credential never reached the engine"
        assert "planet_medres_visual_2024-01_mosaic" in url, f"wrong mosaic: {url}"

    def test_the_credential_is_kept_out_of_the_debug_log(self):
        """cleopatra logs the built URL at DEBUG on a failed fetch, and that URL carries the key.

        Test scenario:
            The assertion is about records, not about how they are stopped: with the logger explicitly at
            DEBUG, nothing it emits during the fetch may escape, and the caller's own configuration must
            be exactly as they left it afterwards — including when the fetch raises, as it does here.
        """
        logger = logging.getLogger("cleopatra.basemap.tiles")
        logger.setLevel(logging.DEBUG)
        try:
            with pytest.raises(ConnectionError):
                self._tropical_map().basemap("Planet.NICFI", preset={"date": "2024-01"})
            assert self.requested, "no fetch was attempted"
            assert not self.escaped, (
                f"a DEBUG record escaped during the fetch: {self.escaped}"
            )
            assert logger.level == logging.DEBUG, "the caller's log level was changed"
            assert not logger.filters, "the suppression outlived the fetch"
        finally:
            logger.setLevel(logging.NOTSET)

    def test_a_non_tropical_domain_is_refused_before_any_fetch(self):
        """The extent guard runs before the engine, so no doomed request is made.

        Test scenario:
            Without it, an out-of-coverage AOI surfaces as cleopatra's ConnectionError after retries — a
            slow, misleading way to learn the basemap does not cover you.
        """
        from digitalearth import Map

        m = Map(domain=TEMPERATE)
        m.ax.set_xlim(TEMPERATE[0], TEMPERATE[2])
        m.ax.set_ylim(TEMPERATE[1], TEMPERATE[3])
        with pytest.raises(ValueError, match="lies entirely outside"):
            m.basemap("Planet.NICFI", preset={"date": "2024-01"})
        assert not self.requested, "a fetch was attempted despite the guard"

    def test_an_ordinary_source_is_unaffected(self):
        """A non-preset source still goes straight to cleopatra, exactly as before.

        Test scenario:
            The dispatch must be additive — every existing basemap() call keeps working.
        """
        with pytest.raises(ConnectionError):
            self._tropical_map().basemap()
        assert self.requested, "the ordinary path no longer reaches the engine"
        assert FAKE_KEY not in self.requested[0], (
            "a credential leaked into a non-keyed basemap"
        )


class TestSourceIsImmutable:
    """``KeyedTileSource`` is frozen — a shared preset must not be mutable by one caller."""

    def test_fields_cannot_be_reassigned(self):
        """The dataclass is frozen, so a registry entry cannot be edited in place.

        Test scenario:
            Presets are built fresh per call today, but the frozen contract is what makes it safe to hand
            the same source to two tiers.
        """
        import dataclasses

        source = planet_nicfi("2024-01")
        with pytest.raises(dataclasses.FrozenInstanceError):
            source.attribution = "mine"

    def test_the_params_mapping_cannot_be_edited_either(self):
        """``frozen=True`` protects the field, not the dict it points at — so the mapping is wrapped.

        Test scenario:
            Without this, ``source.params["mosaic"] = ...`` would edit a preset in place, and presets are
            handed to more than one tier.
        """
        with pytest.raises(TypeError):
            planet_nicfi("2024-01").params["mosaic"] = "something-else"


class TestStaticTierDetails:
    """The parts of the static dispatch not covered by the happy path."""

    @pytest.fixture(autouse=True)
    def _spy(self, monkeypatch):
        """Capture what ``add_tiles`` was handed, without reaching cleopatra at all.

        Args:
            monkeypatch: pytest's patcher.
        """
        pytest.importorskip("matplotlib")
        from digitalearth.static.maps import decoration as static_decoration

        self.calls = []

        def fake_add_tiles(ax, source=None, crs=None, **kwargs):
            self.calls.append({"source": source, "crs": crs, "kwargs": kwargs})
            return "artist"

        monkeypatch.setattr(static_decoration, "add_tiles", fake_add_tiles)
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    @staticmethod
    def _map(domain):
        """A map over ``domain`` with matching axes limits."""
        from digitalearth import Map

        m = Map(domain=domain)
        if domain is not None:
            m.ax.set_xlim(domain[0], domain[2])
            m.ax.set_ylim(domain[1], domain[3])
        return m

    def test_the_attribution_is_forwarded_to_the_engine(self):
        """The service's attribution must reach ``add_tiles`` so the map displays it.

        Test scenario:
            NICFI is non-commercial-only, so its attribution is a licence obligation, not decoration.
        """
        self._map(TROPICAL).basemap("Planet.NICFI", preset={"date": "2024-01"})
        assert self.calls, "add_tiles was never called"
        assert "Planet Labs" in self.calls[0]["kwargs"]["attribution"]

    def test_a_caller_supplied_attribution_is_not_overridden(self):
        """``setdefault`` means an explicit attribution wins, as for any other add_tiles keyword."""
        self._map(TROPICAL).basemap(
            "Planet.NICFI", preset={"date": "2024-01"}, attribution="mine"
        )
        assert self.calls[0]["kwargs"]["attribution"] == "mine"

    def test_the_provider_carries_the_name_and_zoom(self):
        """The built provider keeps the preset's identity and zoom ceiling.

        Test scenario:
            ``max_zoom`` reaches the provider as the ``xyzservices`` metadata it is. cleopatra does not
            read it — it picks a zoom from the extent and caps itself at 19 — so this pins what is
            carried, not a bound that is enforced here. The web tier is where it is honoured; losing it
            there would let MapLibre ask for tiles the
            service does not serve.
        """
        self._map(TROPICAL).basemap("Planet.NICFI", preset={"date": "2024-01"})
        provider = self.calls[0]["source"]
        assert provider.name == "Planet.NICFI.analytic.2024-01", provider.name
        assert provider.max_zoom == 20, provider.max_zoom

    def test_an_explicit_api_key_reaches_the_provider(self):
        """``api_key=`` overrides the environment at the tier boundary too."""
        self._map(TROPICAL).basemap(
            "Planet.NICFI", preset={"date": "2024-01"}, api_key="EXPLICIT-KEY"
        )
        assert "EXPLICIT-KEY" in self.calls[0]["source"].url
        assert FAKE_KEY not in self.calls[0]["source"].url

    def test_a_map_with_nothing_drawn_yet_is_not_guarded(self):
        """With no domain and nothing plotted there is no extent to check, so the basemap goes ahead.

        Test scenario:
            The guard falls back to the axes limits, and before anything is drawn those are matplotlib's
            default unit square — a box that sits inside the NICFI band by coordinates but means nothing.
            Treating it as an extent would decide coverage from a placeholder.
        """
        self._map(None).basemap("Planet.NICFI", preset={"date": "2024-01"})
        assert self.calls, "a map with no extent yet was refused"


class TestUrlSafety:
    """The credential lives in the URL, so nothing may rewrite or expose that URL unnoticed."""

    def test_a_non_http_template_is_refused(self):
        """cleopatra reports a non-http(s) URL by raising *with the URL in the message*.

        Test scenario:
            That message would carry the credential, and it is an exception rather than a log record, so
            the static tier's log suppression does not cover it. Refusing here closes the path.
        """
        source = KeyedTileSource(
            name="Example",
            url_template="ftp://a/{z}/{x}/{y}.png?k={api_key}",
            attribution="Example",
            credential_env="EXAMPLE_KEY",
        )
        with pytest.raises(ValueError, match="non-http"):
            source.tile_url(api_key="K")

    @pytest.mark.parametrize("bad", ["a#b", "a?b", "a&b", "a/b", "a b"])
    def test_a_placeholder_value_cannot_rewrite_the_request(self, bad):
        """A URL delimiter in a substituted value would change what is requested, not fill a slot.

        Args:
            bad: A placeholder value containing a delimiter.

        Test scenario:
            ``#`` is the dangerous one — it truncates the query string and silently drops ``api_key``,
            turning an authenticated request into an anonymous one that fails confusingly.
        """
        with pytest.raises(ValueError, match="rewrite the tile request"):
            planet_nicfi("2024-01", mosaic=bad).tile_url(api_key="K")

    def test_a_normal_mosaic_id_is_unaffected(self):
        """The delimiter guard must not reject the ids Planet actually publishes."""
        url = planet_nicfi("2024-01").tile_url(api_key="K")
        assert "planet_medres_normalized_analytic_2024-01_mosaic" in url


class TestNicfiDateRange:
    """The derived mosaic id is only correct for the months Planet publishes monthly."""

    def test_a_month_before_the_monthly_series_is_refused(self):
        """Pre-2020-09 periods are biannual and named differently, so the derived id cannot exist.

        Test scenario:
            Building it anyway produces a 404 from Planet, which says nothing about the real problem.
        """
        with pytest.raises(ValueError, match="2020-09"):
            planet_nicfi("2019-06")

    def test_an_explicit_mosaic_reaches_the_older_periods(self):
        """``mosaic=`` is the escape hatch for the biannual ids, so the range check must not block it."""
        source = planet_nicfi(
            "2019-06", mosaic="planet_medres_normalized_analytic_2019-06_2019-11_mosaic"
        )
        assert source.params["mosaic"].endswith("2019-06_2019-11_mosaic")


class TestLongitudeConventions:
    """``check_bounds`` compares longitude modulo 360, and rejects extents that are not lon/lat."""

    @pytest.mark.parametrize(
        "extent, refused",
        [
            (
                (200.0, -10.0, 300.0, 10.0),
                False,
            ),  # 0–360 Pacific tropics — same place as -160..-60
            ((-60.0, -5.0, -55.0, 0.0), False),
            ((5.0, 52.0, 6.0, 53.0), True),
        ],
    )
    def test_a_0_360_domain_is_understood(self, extent, refused):
        """A domain expressed 0–360 must not be mistaken for one outside the coverage.

        Args:
            extent: The requested extent.
            refused: Whether the guard should raise.

        Test scenario:
            NICFI spans every longitude, so a tropical box at 200–300°E is inside it — but compared
            literally against ``east <= 180`` it looked outside.
        """
        source = planet_nicfi("2024-01")
        if refused:
            with pytest.raises(ValueError, match="lies entirely outside"):
                source.check_bounds(extent)
        else:
            source.check_bounds(extent)

    @pytest.mark.parametrize(
        "extent, refused",
        [((174.0, -5.0, 176.0, 5.0), False), ((-5.0, -5.0, 5.0, 5.0), True)],
    )
    def test_a_source_crossing_the_antimeridian_is_handled(self, extent, refused):
        """A source spanning 170°E–170°W wraps; it is not an inverted range.

        Args:
            extent: The requested extent.
            refused: Whether the guard should raise.
        """
        source = KeyedTileSource(
            name="Pacific",
            url_template="https://a/{z}/{x}/{y}.png?k={api_key}",
            attribution="Pacific",
            credential_env="EXAMPLE_KEY",
            bounds=(170.0, -10.0, -170.0, 10.0),
        )
        if refused:
            with pytest.raises(ValueError, match="lies entirely outside"):
                source.check_bounds(extent)
        else:
            source.check_bounds(extent)

    def test_a_projected_extent_says_so_rather_than_out_of_coverage(self):
        """Metres are not degrees; reporting "outside coverage" would send the caller the wrong way.

        Test scenario:
            A Web-Mercator extent has latitudes in the millions, which cannot be a lon/lat box.
        """
        with pytest.raises(ValueError, match="not lon/lat"):
            planet_nicfi("2024-01").check_bounds(
                (500000.0, 5800000.0, 510000.0, 5810000.0)
            )

    def test_an_inverted_extent_is_reported_as_inverted(self):
        """South above north is a caller error, not an absence of coverage."""
        with pytest.raises(ValueError, match="inverted"):
            planet_nicfi("2024-01").check_bounds((-60.0, 10.0, -55.0, -10.0))


class TestStaticCoverageGuardSources:
    """Where the static tier gets the extent it checks coverage against (M2), and the preset guard (M5)."""

    @pytest.fixture(autouse=True)
    def _spy(self, monkeypatch):
        """Capture what ``add_tiles`` was handed, without reaching cleopatra.

        Args:
            monkeypatch: pytest's patcher.
        """
        pytest.importorskip("matplotlib")
        from digitalearth.static.maps import decoration as static_decoration

        self.calls = []
        monkeypatch.setattr(
            static_decoration,
            "add_tiles",
            lambda ax, source=None, crs=None, **kw: (
                self.calls.append(source) or "artist"
            ),
        )
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    #: The two AOIs in the display CRS (EPSG:3857 metres), which is what real axes limits hold.
    MERCATOR = {
        "temperate": (556597.0, 6800125.0, 668219.0, 6982997.0),  # the Netherlands
        "tropical": (-6679169.0, -557305.0, -6121300.0, 0.0),  # the Amazon
    }

    @classmethod
    def _map_with_limits(cls, where, domain=None):
        """A map whose axes limits are the named AOI in the display CRS.

        Args:
            where: ``"temperate"`` or ``"tropical"``.
            domain: An optional declared domain, which takes precedence over the limits.

        Returns:
            The configured map.
        """
        from digitalearth import Map

        west, south, east, north = cls.MERCATOR[where]
        m = Map(domain=domain)
        m.ax.set_xlim(west, east)
        m.ax.set_ylim(south, north)
        return m

    def test_the_axes_extent_guards_when_no_domain_was_declared(self):
        """Plot-then-basemap is the common flow, and it declares no domain.

        Test scenario:
            Reading only ``domain`` left the guard inert for it — a Netherlands map happily requested
            NICFI tiles that do not exist there.
        """
        with pytest.raises(ValueError, match="lies entirely outside"):
            self._map_with_limits("temperate").basemap(
                "Planet.NICFI", preset={"date": "2024-01"}
            )
        assert not self.calls, "a fetch was set up despite the guard"

    def test_a_tropical_axes_extent_still_passes(self):
        """The fallback must not refuse a map that is inside the coverage."""
        self._map_with_limits("tropical").basemap(
            "Planet.NICFI", preset={"date": "2024-01"}
        )
        assert self.calls, "a tropical map was refused"

    def test_an_untouched_axes_is_not_mistaken_for_an_extent(self):
        """Matplotlib's default unit square is not a real extent, and must not decide coverage.

        Test scenario:
            (0, 0, 1, 1) sits inside the NICFI band by coordinates, so this cannot be caught by the
            guard passing — it is checked by the basemap being allowed through rather than refused on a
            meaningless box.
        """
        from digitalearth import Map

        Map().basemap("Planet.NICFI", preset={"date": "2024-01"})
        assert self.calls, "an undrawn map was refused on its default limits"

    def test_a_preset_on_an_ordinary_source_is_refused(self):
        """A preset means nothing to cleopatra's default provider, so it is an error (M5).

        Test scenario:
            The static tier passed these straight to add_tiles, which would have raised something far
            less clear.
        """
        with pytest.raises(ValueError, match="no preset keywords"):
            self._map_with_limits("tropical").basemap(preset={"date": "2024-01"})

    def test_a_preset_keyword_written_loose_says_where_it_belongs(self):
        """``**kwargs`` here goes to add_tiles, so a loose ``date=`` would vanish into cleopatra.

        Test scenario:
            All three tiers take a preset as a dict. A caller who writes the keyword loose gets the
            call they meant, not a cleopatra error about an argument it has never heard of.
        """
        with pytest.raises(ValueError, match="not loose"):
            self._map_with_limits("tropical").basemap("Planet.NICFI", date="2024-01")


class TestKeyStaysOutOfTheLog:
    """M3: assert no record escapes, not merely that the logger level was raised."""

    def test_no_log_record_carries_the_credential(self, monkeypatch):
        """A root handler at DEBUG must see no record containing the key.

        Args:
            monkeypatch: pytest's patcher.

        Test scenario:
            The previous test asserted the logger's *level* during the fetch, which would still pass if
            the record reached a handler by another route — propagation to an already-configured root
            logger, for instance.
        """
        pytest.importorskip("matplotlib")
        from cleopatra.basemap import tiles as cleo_tiles

        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

        def failing_fetch(tile, provider, timeout, retries, user_agent="test"):
            logging.getLogger("cleopatra.basemap.tiles").debug(
                "Tile fetch failed for %s",
                provider.build_url(x=tile.x, y=tile.y, z=tile.z),
            )
            raise ConnectionError("network not used in tests")

        monkeypatch.setattr(cleo_tiles, "fetch_single_tile", failing_fetch)

        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        root = logging.getLogger()
        handler = Capture()
        root.addHandler(handler)
        previous_root = root.level
        root.setLevel(logging.DEBUG)
        logging.getLogger("cleopatra.basemap.tiles").setLevel(logging.NOTSET)
        try:
            from digitalearth import Map

            m = Map(domain=TROPICAL)
            m.ax.set_xlim(TROPICAL[0], TROPICAL[2])
            m.ax.set_ylim(TROPICAL[1], TROPICAL[3])
            with pytest.raises(ConnectionError):
                m.basemap("Planet.NICFI", preset={"date": "2024-01"})
        finally:
            root.removeHandler(handler)
            root.setLevel(previous_root)
        leaked = [msg for msg in records if FAKE_KEY in msg]
        assert not leaked, f"the credential reached a log handler: {leaked}"


class TestPresetKeywordErrorsAreNamed:
    """The keyword error names the preset the caller asked for, not the factory behind it (N6)."""

    def test_a_missing_required_keyword_names_the_preset(self):
        """``date`` is required, and the message has to say which preset requires it.

        Test scenario:
            The raw ``TypeError`` reads "planet_nicfi() missing 1 required positional argument", naming a
            function the caller never wrote.
        """
        with pytest.raises(TypeError, match="Planet.NICFI") as err:
            get_keyed_basemap("Planet.NICFI")
        assert "'date'" in str(err.value), (
            f"the keywords it takes are missing: {err.value}"
        )

    def test_a_misspelled_keyword_lists_the_ones_that_exist(self):
        """``dat=`` is a typo, and the answer to a typo is the list of real keywords."""
        with pytest.raises(
            TypeError, match=r"Its keywords are \['date', 'flavour', 'mosaic'\]"
        ):
            get_keyed_basemap("Planet.NICFI", dat="2024-01")

    def test_the_display_name_is_used_even_when_the_caller_lower_cased_it(self):
        """The lookup is case-insensitive, so the message must not echo the caller's spelling back."""
        with pytest.raises(TypeError, match="Planet.NICFI"):
            get_keyed_basemap("planet.nicfi", dat="2024-01")


class TestWhereTheStaticExtentComesFrom:
    """The three ways `_coverage_extent` can end up with nothing to check, and the lon/lat shortcut."""

    @pytest.fixture(autouse=True)
    def _spy(self, monkeypatch):
        """Capture what ``add_tiles`` was handed, without reaching cleopatra.

        Args:
            monkeypatch: pytest's patcher.
        """
        pytest.importorskip("matplotlib")
        from digitalearth.static.maps import decoration as static_decoration

        self.calls = []
        monkeypatch.setattr(
            static_decoration,
            "add_tiles",
            lambda ax, source=None, crs=None, **kw: (
                self.calls.append(source) or "artist"
            ),
        )
        monkeypatch.setenv("PLANET_API_KEY", FAKE_KEY)

    def test_a_domain_that_cannot_be_resolved_leaves_coverage_to_the_service(self):
        """An unknown region name is not a coverage answer, so it must not become a refusal.

        Test scenario:
            ``resolve_domain`` raises ``KeyError`` for a name it does not know. Turning that into
            "outside the coverage" would blame the basemap for a typo in the domain.
        """
        from digitalearth import Map

        Map(domain="atlantis").basemap("Planet.NICFI", preset={"date": "2024-01"})
        assert self.calls, (
            "an unresolvable domain was treated as being outside the coverage"
        )

    def test_lonlat_axes_are_used_as_they_are(self):
        """With a lon/lat display CRS the limits are already degrees — reprojecting them would be wrong.

        Test scenario:
            A Netherlands box in EPSG:4326 is outside the NICFI band, and has to be refused on the
            numbers as they stand.
        """
        from digitalearth import Map

        m = Map(crs=4326)
        m.ax.set_xlim(4.0, 7.0)
        m.ax.set_ylim(51.0, 54.0)
        with pytest.raises(ValueError, match="lies entirely outside"):
            m.basemap("Planet.NICFI", preset={"date": "2024-01"})

    def test_a_crs_that_cannot_be_reprojected_leaves_coverage_to_the_service(
        self, monkeypatch
    ):
        """A CRS pyproj cannot resolve is not a reason to fail the plot.

        Args:
            monkeypatch: pytest's patcher.

        Test scenario:
            ``reproject_coordinates`` raising means the extent is unknown, not that it is uncovered —
            so the request goes ahead and the service answers it.
        """
        from digitalearth import Map
        from digitalearth.static.maps import decoration as static_decoration

        def _refuse(*args, **kwargs):
            """Stand in for a CRS pyproj cannot resolve."""
            raise ValueError("unknown CRS")

        monkeypatch.setattr(static_decoration, "reproject_coordinates", _refuse)
        m = Map()
        m.ax.set_xlim(556597.0, 668219.0)
        m.ax.set_ylim(6800125.0, 6982997.0)
        m.basemap("Planet.NICFI", preset={"date": "2024-01"})
        assert self.calls, (
            "an unreprojectable extent was treated as being outside the coverage"
        )


class TestTheCredentialIsGuardedToo:
    """The credential is the substitution most likely to arrive malformed, and was the one not checked."""

    @pytest.mark.parametrize(
        "bad", ["abc#frag", "abc&extra=1", "abc?x", "abc/d", "a b"]
    )
    def test_a_delimiter_in_the_key_is_refused(self, bad):
        """A `#` in the key truncates the query string, so the request goes out unauthenticated.

        Args:
            bad: A credential containing a URL delimiter.

        Test scenario:
            The service answers that with a 401 that says nothing about the key being malformed, which is
            exactly the confusion the guard exists to prevent for preset values.
        """
        with pytest.raises(ValueError, match="rewrite the tile request"):
            planet_nicfi("2024-01").tile_url(api_key=bad)

    def test_the_error_does_not_echo_the_credential(self):
        """The message may be logged or shown, so it names the placeholder and not the value."""
        with pytest.raises(ValueError) as err:
            planet_nicfi("2024-01").tile_url(api_key="secret#frag")
        assert "secret" not in str(err.value), (
            f"the credential leaked into the message: {err.value}"
        )
        assert "the resolved credential" in str(err.value), str(err.value)

    def test_surrounding_whitespace_is_stripped(self, monkeypatch):
        """``KEY=$(cat key.txt)`` leaves a trailing newline, which is not part of the key.

        Args:
            monkeypatch: pytest's environment patcher.

        Test scenario:
            Without the strip this reaches the delimiter guard as a space and is refused, which would be a
            confusing answer to a key that is otherwise correct.
        """
        monkeypatch.setenv("PLANET_API_KEY", "  abc123\n")
        assert planet_nicfi("2024-01").tile_url().endswith("api_key=abc123")

    def test_a_key_passed_explicitly_is_stripped_as_well(self):
        """The same key read from a file and passed through ``api_key=`` must behave the same way."""
        assert (
            planet_nicfi("2024-01")
            .tile_url(api_key="  abc123\t")
            .endswith("api_key=abc123")
        )


class TestPathSeparatorsAreDelimitersToo:
    """L9: `/` rewrites the path exactly as `?` and `&` rewrite the query."""

    def test_a_mosaic_id_cannot_add_a_path_segment(self):
        """A value with a slash requests a different endpoint, carrying the credential to it.

        Test scenario:
            ``mosaic="../../etc"`` walks up out of the mosaic path entirely; the guard's own wording —
            "rewrite the tile request rather than fill a placeholder" — is exactly what that does.
        """
        with pytest.raises(ValueError, match="rewrite the tile request"):
            planet_nicfi("2024-01", mosaic="../../etc").tile_url(api_key="K")


class TestTheExtentIsValidatedOnItsOwn:
    """Whether an extent is lon/lat is a question about the caller's input, not about the coverage."""

    @staticmethod
    def _global_source():
        """A source that declares no bounds, so the coverage comparison is skipped.

        Returns:
            A :class:`KeyedTileSource` with ``bounds=None``.
        """
        return KeyedTileSource(
            name="Global",
            url_template="https://a/{z}/{x}/{y}.png?k={api_key}",
            attribution="Global",
            credential_env="EXAMPLE_KEY",
        )

    def test_a_source_without_bounds_still_rejects_a_projected_extent(self):
        """Covering the whole globe is not a reason to accept metres.

        Test scenario:
            The early return for ``bounds is None`` used to skip the validation entirely, so garbage
            passed silently for any global source — and the answer would then be wrong the moment a
            preset with real bounds was added.
        """
        with pytest.raises(ValueError, match="not lon/lat"):
            self._global_source().check_bounds(
                (500000.0, 5800000.0, 510000.0, 5810000.0)
            )

    def test_a_projected_extent_whose_northings_look_like_latitudes_is_caught(self):
        """Latitude alone does not settle it — an equatorial Web-Mercator box has small northings.

        Test scenario:
            ``y`` in ``[-50, 50]`` metres passes a latitude range check, and the eastings then read as a
            10 km longitude span, which ``_spans_longitude`` would call "global in longitude". The
            longitude magnitude is what catches it.
        """
        with pytest.raises(ValueError, match="cannot be degrees"):
            planet_nicfi("2024-01").check_bounds((500000.0, -50.0, 510000.0, 50.0))

    def test_an_extent_wider_than_the_planet_is_not_lon_lat(self):
        """No lon/lat box spans more than 360 degrees, whatever convention it is written in."""
        with pytest.raises(ValueError, match="cannot be degrees"):
            planet_nicfi("2024-01").check_bounds((-200.0, -5.0, 200.0, 5.0))

    def test_a_0_360_extent_at_the_top_of_the_range_still_passes(self):
        """The magnitude check must not refuse the 0-360 convention it was written to allow."""
        planet_nicfi("2024-01").check_bounds((340.0, -5.0, 355.0, 5.0))


class TestTheLongitudeAsymmetryIsDeliberate:
    """L8: `south > north` raises while `west > east` does not, and that choice is load-bearing."""

    def test_inverted_longitudes_are_read_as_a_crossing_not_an_error(self):
        """A Pacific extent and a swapped tuple are the same four numbers, and the real case wins.

        Test scenario:
            (10, -5, -10, 5) could be a typo, or a 340-degree box the long way round. Refusing it would
            break every genuine antimeridian extent to catch a mistake the numbers cannot prove.
        """
        planet_nicfi("2024-01").check_bounds((10.0, -5.0, -10.0, 5.0))

    def test_a_fully_swapped_tuple_is_still_caught_by_its_latitudes(self):
        """The latitude half of a swapped tuple is what makes the mistake visible."""
        with pytest.raises(ValueError, match="inverted"):
            planet_nicfi("2024-01").check_bounds((10.0, 5.0, -10.0, -5.0))


class TestASourceCanBeUsedAsAValue:
    """L1: `frozen=True` promises a value type, and a value type that compares equal has to hash."""

    def test_two_identical_sources_are_equal_and_hash_alike(self):
        """Equality without hashing rules out a set, a dict key, and any memoisation.

        Test scenario:
            The generated hash hashes the field tuple, and ``params`` is a ``mappingproxy`` that
            delegates to the dict underneath — so it raised ``TypeError: unhashable type: 'dict'``.
        """
        first, second = planet_nicfi("2024-01"), planet_nicfi("2024-01")
        assert first == second, "two sources built from the same preset differ"
        assert hash(first) == hash(second), "equal sources hash differently"
        assert len({first, second}) == 1, "equal sources did not collapse in a set"

    def test_sources_that_differ_in_a_preset_value_do_not_collide(self):
        """The mosaic is part of what a source *is*, so it has to reach the hash."""
        assert planet_nicfi("2024-01") != planet_nicfi("2024-02")
        assert len({planet_nicfi("2024-01"), planet_nicfi("2024-02")}) == 2

    def test_a_source_works_as_a_dict_key(self):
        """The point of hashing is being usable as a key, so use one."""
        cache = {planet_nicfi("2024-01"): "tiles"}
        assert cache[planet_nicfi("2024-01")] == "tiles"


class TestOnlyKeywordProblemsAreRenamed:
    """L2: the preset-keyword wrapper must not swallow a TypeError raised by the factory body."""

    def test_a_type_error_from_inside_the_factory_is_not_relabelled(self):
        """``date=5`` is a wrong *type*, not a wrong keyword, and the message has to say so.

        Test scenario:
            The blanket handler reported "expected string or bytes-like object, got 'int'" followed by a
            keyword list that was already correct, burying the real problem behind a wrong suggestion.
        """
        with pytest.raises(TypeError) as err:
            get_keyed_basemap("Planet.NICFI", date=5)
        assert "Its keywords are" not in str(err.value), (
            f"a type error was renamed: {err.value}"
        )

    def test_a_keyword_problem_is_still_renamed(self):
        """The rename is right for the case it was written for, and must survive the narrowing."""
        with pytest.raises(TypeError, match="Its keywords are"):
            get_keyed_basemap("Planet.NICFI", dat="2024-01")

    def test_the_renamed_message_reads_as_two_sentences(self):
        """The preset name, the binding error and the keyword list ran together without punctuation."""
        with pytest.raises(TypeError) as err:
            get_keyed_basemap("Planet.NICFI", dat="2024-01")
        assert ". Its keywords are" in str(err.value), str(err.value)


class TestThePresetKeywordSetIsDerived:
    """M3: the backends tell a preset keyword from one of their own; that set must follow the registry."""

    def test_it_matches_what_the_registered_presets_accept(self):
        """A literal repeated in two backends drifts the moment a second preset is added.

        Test scenario:
            The set is built from the factory signatures, so a preset with a new keyword is understood by
            every backend without either of them being edited.
        """
        import inspect

        from digitalearth.base.basemaps import PRESET_KEYWORDS

        expected = {
            keyword
            for factory in KEYED_BASEMAPS.values()
            for keyword in inspect.signature(factory).parameters
        }
        assert PRESET_KEYWORDS == expected, f"{PRESET_KEYWORDS} != {expected}"

    def test_the_static_backend_uses_the_shared_set(self):
        """Importing it is what makes the guarantee hold; a local copy would not.

        Test scenario:
            The static tier is the one that still needs the set: its ``**kwargs`` go to ``add_tiles``, so
            a preset keyword written loose there has to be recognised and refused. The web and
            interactive tiers take the preset as an explicit dict and need no such test.
        """
        from digitalearth.base.basemaps import PRESET_KEYWORDS
        from digitalearth.static.maps import decoration as static_decoration

        assert static_decoration.PRESET_KEYWORDS is PRESET_KEYWORDS


class TestTheEnvelopeIsReprojectedAsAShape:
    """M1/M2: outside the cylindrical projections, two corners do not describe a projected rectangle."""

    @pytest.fixture(autouse=True)
    def _need_matplotlib(self):
        """Skip when matplotlib is absent."""
        pytest.importorskip("matplotlib")

    def test_a_polar_box_yields_a_real_envelope(self):
        """A pole-centred box has no meaningful "SW corner", so the corner pair described nothing.

        Test scenario:
            Arctic Polar Stereographic over a 2000 km box gave (-45, 77.04, 135, 77.04) — a zero-height
            box at one latitude. Sampling the edges gives a span in both axes.
        """
        from digitalearth import Map

        m = Map(crs=3995)
        m.ax.set_xlim(-1_000_000, 1_000_000)
        m.ax.set_ylim(-1_000_000, 1_000_000)
        west, south, east, north = m._axes_lonlat_extent()
        assert north > south, (
            f"the envelope is still flat: {(west, south, east, north)}"
        )
        assert east - west > 180.0, (
            f"a polar box spans most meridians, got {west}..{east}"
        )
        assert south > 60.0, f"an Arctic box cannot reach {south}"

    def test_a_web_mercator_box_is_unchanged(self):
        """Sampling must not move the answer for the cylindrical case, where corners were already right."""
        from digitalearth import Map

        m = Map(crs=3857)
        m.ax.set_xlim(556597.0, 668219.0)
        m.ax.set_ylim(6800125.0, 6982997.0)
        west, south, east, north = m._axes_lonlat_extent()
        assert (round(west), round(south), round(north)) == (5, 52, 53), (
            west,
            south,
            north,
        )

    def test_a_projection_whose_corners_have_no_lon_lat_fails_open(self):
        """pyproj answers `inf` instead of raising, and an unknown extent must not become a refusal.

        Test scenario:
            An orthographic globe's box corners are off the visible hemisphere. The guard used to pass
            (inf, inf, inf, inf) to check_bounds, which reported it as "not lon/lat" — blaming the caller
            for an extent they never supplied.
        """
        from digitalearth import Map

        m = Map(crs="+proj=ortho +lat_0=0 +lon_0=0")
        m.ax.set_xlim(-6.4e6, 6.4e6)
        m.ax.set_ylim(-6.4e6, 6.4e6)
        assert m._axes_lonlat_extent() is None, (
            "a non-finite reprojection was treated as an extent"
        )


class TestACredentialWithNothingToAuthenticate:
    """L6: a caller who passes api_key believes they are authenticating; silence would be wrong."""

    @pytest.fixture(autouse=True)
    def _need_matplotlib(self, monkeypatch):
        """Skip without matplotlib, and keep cleopatra out of it.

        Args:
            monkeypatch: pytest's patcher.
        """
        pytest.importorskip("matplotlib")
        from digitalearth.static.maps import decoration as static_decoration

        monkeypatch.setattr(
            static_decoration,
            "add_tiles",
            lambda ax, source=None, crs=None, **kw: "artist",
        )

    def test_an_api_key_on_a_token_free_source_is_refused(self):
        """The parameter is consumed by the signature now, so nothing downstream would complain.

        Test scenario:
            Before the keyed presets, api_key= reached add_tiles and failed there. Hoisting it into a
            keyword-only parameter made it silently vanish — two lines from a stray date= being a hard
            error for exactly the same reason.
        """
        from digitalearth import Map

        with pytest.raises(ValueError, match="takes no api_key"):
            Map(domain=TROPICAL).basemap(api_key="x")

    def test_a_keyed_preset_still_takes_one(self):
        """The guard must not reach the case the parameter exists for."""
        from digitalearth import Map

        m = Map(domain=TROPICAL)
        m.ax.set_xlim(TROPICAL[0], TROPICAL[2])
        m.ax.set_ylim(TROPICAL[1], TROPICAL[3])
        m.basemap("Planet.NICFI", preset={"date": "2024-01"}, api_key=FAKE_KEY)
