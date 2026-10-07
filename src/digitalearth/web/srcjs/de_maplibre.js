/*
 * de_maplibre.js — Digital-Earth's own standalone-HTML runtime for the web tier.
 *
 * Why this exists (WB-13/17/20). py-maplibregl's own `to_html` renders the map through a sealed,
 * minified bundle that hardcodes a single container id (`#pymaplibregl`), never exposes the map
 * object, and keeps maplibre-gl / its custom controls internal. That makes a split/swipe export,
 * a synced minimap, and a live on-map measure readout impossible, because each needs the map
 * object and/or a second map in one page. This runtime rebuilds the map from the SAME serialized
 * state py-maplibregl records — `{mapOptions, calls}` from `Map.to_dict()` — against a container we
 * own, exposes the map on `window.DE.maps[id]`, and can host more than one map in a page, so the
 * export document can wire features across them.
 *
 * It reads the serialized `calls` list, each item `[method, [arg, ...]]`, and replays it onto a
 * real `maplibregl.Map` loaded from CDN. Most calls are a plain `map[method](...args)`; a handful
 * (controls, popups, fit-bounds) are interpreted here. The custom controls py-maplibregl ships in
 * its bundle (the layer switcher, the info box) are reimplemented as small maplibre `IControl`s so
 * a page exported through this runtime keeps the switcher the widget path has.
 */
(function (global) {
  "use strict";

  const DE = (global.DE = global.DE || { maps: {}, version: 1 });

  /* ------------------------------------------------------------------ controls */

  // A checkbox layer switcher — the reimplementation of py-maplibregl's LayerSwitcherControl. It
  // toggles each named layer's `visibility` layout property, which is exactly what the widget's
  // control does, so a saved page keeps the show/hide rows the step picker and `layer_control` add.
  class DELayerSwitcher {
    constructor(options) {
      this._layerIds = (options && options.layerIds) || [];
      this._theme = (options && options.theme) || "default";
    }
    onAdd(map) {
      this._map = map;
      const el = (this._container = document.createElement("div"));
      el.className =
        "maplibregl-ctrl maplibregl-ctrl-group de-layer-switcher de-theme-" +
        this._theme;
      for (const id of this._layerIds) {
        const row = document.createElement("label");
        row.className = "de-ls-row";
        const box = document.createElement("input");
        box.type = "checkbox";
        // A layer built hidden must read unchecked; "none" is the only hidden value.
        box.checked = map.getLayer(id)
          ? map.getLayoutProperty(id, "visibility") !== "none"
          : true;
        box.addEventListener("change", () => {
          if (!map.getLayer(id)) return;
          map.setLayoutProperty(
            id,
            "visibility",
            box.checked ? "visible" : "none"
          );
        });
        row.appendChild(box);
        row.appendChild(document.createTextNode(" " + id));
        el.appendChild(row);
      }
      return el;
    }
    onRemove() {
      if (this._container && this._container.parentNode) {
        this._container.parentNode.removeChild(this._container);
      }
      this._map = undefined;
    }
  }

  // A static HTML box pinned to a corner — the reimplementation of py-maplibregl's InfoBoxControl,
  // used by the tier for a legend / title panel.
  class DEInfoBox {
    constructor(options) {
      this._content = (options && (options.content || options.cssText)) || "";
      this._css = (options && options.cssText) || "";
    }
    onAdd() {
      const el = (this._container = document.createElement("div"));
      el.className = "maplibregl-ctrl de-infobox";
      if (this._css) el.setAttribute("style", this._css);
      el.innerHTML = this._content;
      return el;
    }
    onRemove() {
      if (this._container && this._container.parentNode) {
        this._container.parentNode.removeChild(this._container);
      }
    }
  }

  // A place-search box — the reimplementation of py-maplibregl's MapTilerGeocodingControl. Rather
  // than vendor the upstream geocoding-control bundle, it calls MapTiler's geocoding API directly
  // with the caller's key and flies/fits to a chosen result, which is all the tier's `geocoder()`
  // asks for and keeps the control small and dependency-free.
  class DEGeocoder {
    constructor(options) {
      options = options || {};
      this._key = options.apiKey;
      this._placeholder = options.placeholder || "Search";
    }
    onAdd(map) {
      this._map = map;
      const el = (this._container = document.createElement("div"));
      el.className = "maplibregl-ctrl maplibregl-ctrl-group de-geocoder";
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = this._placeholder;
      input.className = "de-geocoder-input";
      const results = document.createElement("div");
      results.className = "de-geocoder-results";
      el.appendChild(input);
      el.appendChild(results);
      const self = this;
      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") self._search(input.value, results);
      });
      return el;
    }
    _search(query, results) {
      if (!query || !this._key) return;
      const map = this._map;
      const url =
        "https://api.maptiler.com/geocoding/" +
        encodeURIComponent(query) +
        ".json?key=" +
        encodeURIComponent(this._key) +
        "&limit=5";
      fetch(url)
        .then((r) => r.json())
        .then((data) => {
          results.innerHTML = "";
          (data.features || []).forEach((f) => {
            const row = document.createElement("div");
            row.className = "de-geocoder-row";
            row.textContent = f.place_name || f.text || "";
            row.addEventListener("click", () => {
              if (f.bbox) {
                map.fitBounds([
                  [f.bbox[0], f.bbox[1]],
                  [f.bbox[2], f.bbox[3]],
                ]);
              } else if (f.center) {
                map.flyTo({ center: f.center, zoom: 12 });
              }
              results.innerHTML = "";
            });
            results.appendChild(row);
          });
        })
        .catch(() => {
          results.textContent = "search failed";
        });
    }
    onRemove() {
      if (this._container && this._container.parentNode) {
        this._container.parentNode.removeChild(this._container);
      }
      this._map = undefined;
    }
  }

  // The standard maplibre-gl controls, by the class name py-maplibregl serializes. Each is a real
  // `maplibregl.*Control`, constructed with the options dict the widget recorded.
  const STANDARD_CONTROLS = {
    NavigationControl: true,
    ScaleControl: true,
    FullscreenControl: true,
    GeolocateControl: true,
    AttributionControl: true,
    GlobeControl: true,
    TerrainControl: true,
  };

  function makeControl(type, options) {
    if (type === "LayerSwitcherControl") return new DELayerSwitcher(options);
    if (type === "InfoBoxControl") return new DEInfoBox(options);
    if (type === "MapTilerGeocodingControl") return new DEGeocoder(options);
    if (STANDARD_CONTROLS[type] && global.maplibregl[type]) {
      return new global.maplibregl[type](options || {});
    }
    // An unknown control is reported, not silently dropped — the export would otherwise look
    // complete while quietly missing a control the caller asked for.
    throw new Error("de_maplibre: unsupported control " + type);
  }

  /* -------------------------------------------------------------- popups/tooltips */

  // Escape a value for safe insertion into popup HTML. A popup is built as an HTML string and set
  // with `setHTML` (which assigns `innerHTML`), so a feature property containing markup would render
  // as live HTML — the same injection class as the embedded map state. Keys and values are escaped.
  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function propertyHtml(feature, field) {
    const props = (feature && feature.properties) || {};
    if (field && Object.prototype.hasOwnProperty.call(props, field)) {
      return escapeHtml(field) + ": " + escapeHtml(props[field]);
    }
    return Object.keys(props)
      .map((k) => escapeHtml(k) + ": " + escapeHtml(props[k]))
      .join("<br>");
  }

  function bindPopup(map, layerId, field, onHover) {
    const Popup = global.maplibregl.Popup;
    if (onHover) {
      const popup = new Popup({ closeButton: false, closeOnClick: false });
      map.on("mousemove", layerId, (e) => {
        map.getCanvas().style.cursor = "pointer";
        const f = e.features && e.features[0];
        if (f) popup.setLngLat(e.lngLat).setHTML(propertyHtml(f, field)).addTo(map);
      });
      map.on("mouseleave", layerId, () => {
        map.getCanvas().style.cursor = "";
        popup.remove();
      });
    } else {
      map.on("click", layerId, (e) => {
        const f = e.features && e.features[0];
        if (f) {
          new Popup().setLngLat(e.lngLat).setHTML(propertyHtml(f, field)).addTo(map);
        }
      });
    }
  }

  /* ----------------------------------------------------------------- call replay */

  // Methods replayed verbatim as `map[name](...args)` — these are maplibre-gl's own, and the widget
  // records them in the shape maplibre expects.
  const PASSTHROUGH = {
    addSource: true,
    addLayer: true,
    setPaintProperty: true,
    setLayoutProperty: true,
    setFilter: true,
    setSourceData: true,
    setSky: true,
    setProjection: true,
    setLight: true,
  };

  // The one call the digitalearth web tier never emits — it draws circle layers, not `Marker`s — so
  // it is refused by name rather than reimplemented against nothing. A refusal (not a silent drop)
  // means a future map that did add a marker would fail loudly here instead of exporting it missing.
  const UNSUPPORTED = {
    addMarker: "markers",
  };

  // Refuse any deck.gl `@@` accessor, at any depth — a string value beginning with `@@` (e.g.
  // `"@@=properties.v"`) or an object key beginning with `@@` (e.g. `{"@@function": …}`). Because the
  // layers are built directly (not through deck's JSONConverter), such an accessor would be copied
  // through and silently mis-render, so it is refused loudly instead. The GeoJSON `data` is not
  // scanned — it is feature data, not deck config, and a property value that happens to start with
  // `@@` is not an accessor.
  function refuseDeckAccessors(value) {
    if (typeof value === "string") {
      if (value.indexOf("@@") === 0) {
        throw new Error("de_maplibre: deck.gl accessor expression not supported: " + value);
      }
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) refuseDeckAccessors(item);
      return;
    }
    if (value && typeof value === "object") {
      for (const key in value) {
        if (key.indexOf("@@") === 0) {
          throw new Error("de_maplibre: deck.gl accessor key not supported: " + key);
        }
        refuseDeckAccessors(value[key]);
      }
    }
  }

  // deck.gl layers are serialized in deck's `@@type` JSON. The tier's specs use only plain values —
  // the accessors (getFillColor, getPointRadius, …) are constants, never `@@function`/`@@=` closures —
  // so each layer is built directly from its class in the deck.gl standalone bundle (which ships the
  // layers and `MapboxOverlay` but not the separate `@deck.gl/json` converter). A `MapboxOverlay`
  // draws the layers over the map; the overlay is kept on the map so a later `setDeckLayers` can swap
  // them. Any `@@` accessor (at any depth, outside the GeoJSON data) is refused rather than passed
  // through — see `refuseDeckAccessors`.
  function deckLayersFrom(specs) {
    const deck = global.deck;
    return (specs || []).map(function (spec) {
      const type = spec["@@type"];
      const Layer = deck[type];
      if (!Layer) throw new Error("de_maplibre: unknown deck.gl layer " + type);
      const props = {};
      for (const key in spec) {
        if (key === "@@type") continue;
        if (key !== "data") refuseDeckAccessors(spec[key]);
        props[key] = spec[key];
      }
      return new Layer(props);
    });
  }

  function applyCall(map, call) {
    const name = call[0];
    const args = call[1] || [];
    if (PASSTHROUGH[name]) {
      map[name].apply(map, args);
      return;
    }
    if (name === "addControl") {
      const [type, options, position] = args;
      map.addControl(makeControl(type, options), position || "top-right");
      return;
    }
    if (name === "fitBounds") {
      // py-maplibregl records a flat [w, s, e, n]; maplibre wants [[w, s], [e, n]].
      const [b, options] = args;
      map.fitBounds(
        [
          [b[0], b[1]],
          [b[2], b[3]],
        ],
        options || {}
      );
      return;
    }
    if (name === "addPopup" || name === "addTooltip") {
      const [layerId, field] = args;
      bindPopup(map, layerId, field || null, name === "addTooltip");
      return;
    }
    if (name === "addDeckOverlay") {
      const overlay = new global.deck.MapboxOverlay({
        interleaved: false,
        layers: deckLayersFrom(args[0]),
      });
      map.addControl(overlay);
      map._deOverlay = overlay;
      return;
    }
    if (name === "setDeckLayers") {
      if (!map._deOverlay) {
        throw new Error(
          "de_maplibre: setDeckLayers with no deck overlay on the map (addDeckOverlay must come first)"
        );
      }
      map._deOverlay.setProps({ layers: deckLayersFrom(args[0]) });
      return;
    }
    if (name === "addMapboxDraw") {
      const draw = new global.MapboxDraw(args[0] || {});
      map.addControl(draw);
      map._deDraw = draw;
      return;
    }
    if (UNSUPPORTED[name]) {
      throw new Error(
        "de_maplibre: the standalone export does not support " +
          UNSUPPORTED[name] +
          " yet (call " +
          name +
          ")"
      );
    }
    // Anything else: the widget's own fall-through contract — invoke it on the map if it exists.
    if (typeof map[name] === "function") {
      map[name].apply(map, args);
      return;
    }
    throw new Error("de_maplibre: unknown call " + name);
  }

  function replay(map, calls) {
    for (const call of calls || []) applyCall(map, call);
  }

  /* --------------------------------------------------------------- map building */

  // Build one map from its serialized `{mapOptions, calls}` into the container with id `containerId`,
  // register it on `window.DE.maps[containerId]`, and return it. `onReady(map)` runs once the style
  // has loaded and the calls have been replayed — which is where a feature wires itself.
  DE.buildMap = function (containerId, data, onReady) {
    const options = Object.assign({ container: containerId }, data.mapOptions || {});
    const map = new global.maplibregl.Map(options);
    DE.maps[containerId] = map;
    map.on("load", function () {
      replay(map, data.calls || []);
      if (typeof onReady === "function") onReady(map);
    });
    return map;
  };

  DE.DELayerSwitcher = DELayerSwitcher;
  DE.DEInfoBox = DEInfoBox;
  // Exposed so a test can check popup content is HTML-escaped without having to fire a map event.
  DE.propertyHtml = propertyHtml;
  DE.escapeHtml = escapeHtml;
})(typeof window !== "undefined" ? window : this);
