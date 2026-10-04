# Keyed basemaps

Most basemaps in `digitalearth` need nothing from you — `Map().basemap()`, `WebMap().basemap("CartoDark")` and
`InteractiveMap().tiles("CartoLight")` all draw token-free public tiles. A **keyed** basemap is one the service
will not serve without a credential, so it needs two things you have to know in advance: an environment variable
to set, and the preset's own keywords.

## Planet NICFI

[Planet's NICFI basemaps](https://www.planet.com/nicfi/) are ~4.77 m monthly mosaics of the tropics, free for
**non-commercial use** under the NICFI terms. They are the first keyed preset.

```bash
export PLANET_API_KEY="…"        # Windows PowerShell: $env:PLANET_API_KEY = "…"
```

```python
from digitalearth import Map

m = Map(domain=(-60, -5, -55, 0))          # somewhere in the Amazon
m.basemap("Planet.NICFI", preset={"date": "2024-01"})
```

Every tier takes the preset the same way — as a `preset` dict, because on the static tier `**kwargs` belongs
to cleopatra and on the interactive tier it belongs to HoloViews' style options:

```python
from digitalearth.web import WebMap
from digitalearth.interactive import InteractiveMap

WebMap().basemap("Planet.NICFI", preset={"date": "2024-01"})
InteractiveMap().tiles("Planet.NICFI", preset={"date": "2024-01"})
```

### Keywords

- **`date`** (required) — the mosaic month, `YYYY-MM`. The monthly series starts at **2020-09**; earlier
  periods are biannual and need an explicit `mosaic`.
- **`flavour`** — `"analytic"` (default, surface reflectance) or `"visual"`. These are different Planet
  products, not a styling flag.
- **`mosaic`** — an explicit Planet mosaic id, overriding the one derived from `date` and `flavour`.

Write them in the `preset` dict, not loose: `basemap("Planet.NICFI", date="2024-01")` is refused with a
message pointing at the right form.

`api_key=` overrides the environment variable for one call. Passing it to a basemap that needs no credential
is an error rather than a no-op, so a call that looks authenticated always is.

### Coverage

NICFI publishes roughly 30°N–30°S. The **static** tier checks the extent it is about to draw against that band
and refuses a basemap that would render blank after a round of failed fetches — it renders one fixed extent, so
an uncovered basemap is a dead end. The web and interactive tiers do not check: their maps are pannable, so the
opening view is not where the viewer stays.

## Treat a saved map as a secret

The credential is part of the tile URL. That is how an XYZ service authenticates a browser, so it is inherent
rather than a bug — but it has consequences:

- **A saved web map contains the key.** `WebMap.basemap("Planet.NICFI").save("map.html")` writes the tile URL,
  key and all, into the HTML. So does a rendered interactive map, which means a committed notebook with saved
  output carries it too. Do not commit, publish, or paste such a file.
- **The static tier does not have this exposure.** It fetches tiles at plot time and embeds only the resulting
  image. It also silences the one cleopatra log record that would carry the URL.
- **Do not run tile fetches at `DEBUG` into shared logs.**

## OGC services: WMS and WMTS

The static tier tiles an OGC service through cleopatra's provider objects, passed to `basemap()` as they are.
`WMSProvider` requests one `GetMap` per Web Mercator tile; `WMTSProvider` requests `GetTile` on the service's
tile matrix set, by key-value parameters or by filling a RESTful URL template.

```python
from cleopatra.basemap.ogc import WMSProvider, WMTSProvider
from digitalearth import Map

m = Map(crs=3857)
m.set_domain((3.3, 50.7, 7.3, 53.6))       # tiles cover what the axes shows
m.basemap(WMSProvider("https://example.org/geoserver/wms", "topp:states"))
m.basemap(WMTSProvider("https://example.org/wmts/{TileMatrix}/{TileRow}/{TileCol}.png", "basemap"))
```

A credential the service needs goes in the provider's `extra_params`.

### What a figure records of an OGC basemap

The provider **object** is held beside the layer — it is an engine object, and it carries whatever is in
`extra_params`. What the figure's description carries is the provider's own fields, every one of them except
`extra_params`:

```python
m.figure_spec.layers.get("bm").symbology.props["source"]
# {'ogc': 'wms', 'url': 'https://example.org/geoserver/wms', 'layers': 'topp:states',
#  'styles': '', 'version': '1.3.0', 'image_format': 'image/png', 'transparent': True,
#  'tile_size': 256, 'attribution': ''}
```

That is enough to rebuild the provider, so a figure saved to JSON and drawn again on this tier — through
`Map.from_figure`, or `to_backend(spec, "matplotlib")` — asks the **same service** for the same layer. Before
this was recorded, such a figure named no service at all and quietly drew the shared default basemap in its
place.

Replaying it on *another* tier is a separate matter and does not work today: `WebMap.from_figure` refuses a
static `basemap` layer outright — `layer <id> (basemap) cannot be drawn by the web tier: its symbology records
none of ['opacity']` — and it does so whether the source was an OGC provider or a plain provider name. That is
the cross-tier basemap gap, not an OGC one.

`extra_params` deliberately does not travel, for the same reason an `xyzservices` `apikey` does not: a figure
is written to JSON and read back, and a token written into one leaks with it. So a replayed figure asks the
service **without** a credential. If the service needs one, it answers with its own error — which is the
honest outcome; pass the provider again, with its `extra_params`, to draw it authenticated.

## Adding a preset

Presets live in [`digitalearth.base.basemaps`][digitalearth.base.basemaps] — engine-neutral definitions (URL
template, credential variable, attribution, coverage) that every backend resolves by name. A new one is a
factory returning a `KeyedTileSource`, registered in `KEYED_BASEMAPS` with its display spelling in
`KEYED_BASEMAP_NAMES`. The backends pick up its keywords automatically.

::: digitalearth.base.basemaps
    options:
      members:
        - KeyedTileSource
        - planet_nicfi
        - get_keyed_basemap
        - is_keyed_basemap
