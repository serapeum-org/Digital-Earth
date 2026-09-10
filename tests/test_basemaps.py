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
        self.levels = []

        def spy(tile, provider, timeout, retries, user_agent="test"):
            self.requested.append(provider.build_url(x=tile.x, y=tile.y, z=tile.z))
            self.levels.append(logging.getLogger("cleopatra.basemap.tiles").level)
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
                "Planet.NICFI", date="2024-01", flavour="visual"
            )
        assert self.requested, "no tile fetch was attempted"
        url = self.requested[0]
        assert FAKE_KEY in url, "the credential never reached the engine"
        assert "planet_medres_visual_2024-01_mosaic" in url, f"wrong mosaic: {url}"

    def test_the_credential_is_kept_out_of_the_debug_log(self):
        """cleopatra logs the built URL at DEBUG on a failed fetch, and that URL carries the key.

        Test scenario:
            The fetch is wrapped so that logger sits above DEBUG for its duration, and is restored after —
            including when the fetch raises, which is what happens here.
        """
        logger = logging.getLogger("cleopatra.basemap.tiles")
        logger.setLevel(logging.DEBUG)
        try:
            with pytest.raises(ConnectionError):
                self._tropical_map().basemap("Planet.NICFI", date="2024-01")
            assert self.levels, "no fetch was attempted"
            assert all(lvl > logging.DEBUG for lvl in self.levels), (
                f"the keyed URL could have been logged at DEBUG: levels {self.levels}"
            )
            assert logger.level == logging.DEBUG, (
                "the caller's log level was not restored"
            )
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
            m.basemap("Planet.NICFI", date="2024-01")
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
        self._map(TROPICAL).basemap("Planet.NICFI", date="2024-01")
        assert self.calls, "add_tiles was never called"
        assert "Planet Labs" in self.calls[0]["kwargs"]["attribution"]

    def test_a_caller_supplied_attribution_is_not_overridden(self):
        """``setdefault`` means an explicit attribution wins, as for any other add_tiles keyword."""
        self._map(TROPICAL).basemap("Planet.NICFI", date="2024-01", attribution="mine")
        assert self.calls[0]["kwargs"]["attribution"] == "mine"

    def test_the_provider_carries_the_name_and_zoom(self):
        """The built provider keeps the preset's identity and zoom ceiling.

        Test scenario:
            ``max_zoom`` bounds what cleopatra will request; losing it would let it ask for tiles the
            service does not serve.
        """
        self._map(TROPICAL).basemap("Planet.NICFI", date="2024-01")
        provider = self.calls[0]["source"]
        assert provider.name == "Planet.NICFI.analytic.2024-01", provider.name
        assert provider.max_zoom == 20, provider.max_zoom

    def test_an_explicit_api_key_reaches_the_provider(self):
        """``api_key=`` overrides the environment at the tier boundary too."""
        self._map(TROPICAL).basemap(
            "Planet.NICFI", date="2024-01", api_key="EXPLICIT-KEY"
        )
        assert "EXPLICIT-KEY" in self.calls[0]["source"].url
        assert FAKE_KEY not in self.calls[0]["source"].url

    def test_a_map_without_a_domain_is_not_guarded(self):
        """With no declared domain there is nothing to check, so the basemap must still be allowed.

        Test scenario:
            The guard reads the map's ``domain``, not the axes limits — which before anything is drawn are
            matplotlib's default unit square and would refuse every keyed basemap on Earth.
        """
        self._map(None).basemap("Planet.NICFI", date="2024-01")
        assert self.calls, "an undomained map was refused"


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

    @pytest.mark.parametrize("bad", ["a#b", "a?b", "a&b", "a b"])
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
                m.basemap("Planet.NICFI", date="2024-01")
        finally:
            root.removeHandler(handler)
            root.setLevel(previous_root)
        leaked = [msg for msg in records if FAKE_KEY in msg]
        assert not leaked, f"the credential reached a log handler: {leaked}"
