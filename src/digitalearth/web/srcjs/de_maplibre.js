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
    if (STANDARD_CONTROLS[type] && global.maplibregl[type]) {
      return new global.maplibregl[type](options || {});
    }
    // An unknown control is reported, not silently dropped — the export would otherwise look
    // complete while quietly missing a control the caller asked for.
    throw new Error("de_maplibre: unsupported control " + type);
  }

  /* -------------------------------------------------------------- popups/tooltips */

  function propertyHtml(feature, field) {
    const props = (feature && feature.properties) || {};
    if (field && Object.prototype.hasOwnProperty.call(props, field)) {
      return String(field) + ": " + String(props[field]);
    }
    return Object.keys(props)
      .map((k) => String(k) + ": " + String(props[k]))
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

  // Methods this runtime does not yet rebuild. They are refused by name rather than dropped, so a
  // page is never exported silently missing a deck.gl overlay, a draw control added for its own
  // sake, or a marker. (A measure readout adds its own draw control through the feature layer, not
  // through this list.)
  const UNSUPPORTED = {
    addDeckOverlay: "deck.gl overlays",
    setDeckLayers: "deck.gl overlays",
    addMarker: "markers",
  };

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
})(typeof window !== "undefined" ? window : this);
