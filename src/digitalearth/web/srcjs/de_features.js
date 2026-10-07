/*
 * de_features.js — cross-map features for the Digital-Earth standalone export.
 *
 * These are the capabilities py-maplibregl's single-container export cannot carry, because they
 * need more than one map in a page and/or a handle on the live map object. They run on top of the
 * maps `de_maplibre.js` built and exposed on `window.DE.maps`.
 *
 *   DE.swipe   — WB-13: two maps overlaid, the "after" one clipped by a draggable divider, their
 *                cameras kept in sync, so a viewer wipes between two states of the same area.
 */
(function (global) {
  "use strict";

  const DE = (global.DE = global.DE || { maps: {}, version: 1 });

  // Keep two maps' cameras identical. A move on either is mirrored onto the other; the `syncing`
  // guard stops the mirrored jump from firing a second move and looping.
  function linkCameras(a, b) {
    let syncing = false;
    function mirror(from, to) {
      if (syncing) return;
      syncing = true;
      to.jumpTo({
        center: from.getCenter(),
        zoom: from.getZoom(),
        bearing: from.getBearing(),
        pitch: from.getPitch(),
      });
      syncing = false;
    }
    a.on("move", () => mirror(a, b));
    b.on("move", () => mirror(b, a));
    // Seed the follower from the leader so the two start aligned even before the first move.
    mirror(a, b);
  }

  // WB-13 — wipe between `beforeId` and `afterId` (both already built by DE.buildMap). The after
  // map is clipped to the right of the divider; dragging the divider (or the pointer across the
  // container) moves the split. `wrapId` is the positioned wrapper holding both map containers.
  DE.swipe = function (wrapId, beforeId, afterId) {
    const before = DE.maps[beforeId];
    const after = DE.maps[afterId];
    const wrap = document.getElementById(wrapId);
    const afterEl = document.getElementById(afterId);
    if (!before || !after || !wrap || !afterEl) {
      throw new Error("de_maplibre: swipe needs two built maps in a wrapper");
    }
    linkCameras(before, after);

    const divider = document.createElement("div");
    divider.className = "de-swipe-divider";
    divider.setAttribute("role", "separator");
    divider.setAttribute("aria-label", "swipe divider");
    wrap.appendChild(divider);

    let fraction = 0.5;
    function place() {
      const pct = Math.max(0, Math.min(1, fraction)) * 100;
      // Clip the "after" map so only the portion right of the divider shows; the "before" map
      // shows through on the left.
      afterEl.style.clipPath = "inset(0 0 0 " + pct + "%)";
      afterEl.style.webkitClipPath = "inset(0 0 0 " + pct + "%)";
      divider.style.left = pct + "%";
    }
    place();

    let dragging = false;
    function fractionFromEvent(ev) {
      const rect = wrap.getBoundingClientRect();
      const x = (ev.touches ? ev.touches[0].clientX : ev.clientX) - rect.left;
      return rect.width ? x / rect.width : 0.5;
    }
    function onDown(ev) {
      dragging = true;
      fraction = fractionFromEvent(ev);
      place();
      ev.preventDefault();
    }
    function onMove(ev) {
      if (!dragging) return;
      fraction = fractionFromEvent(ev);
      place();
    }
    function onUp() {
      dragging = false;
    }
    divider.addEventListener("mousedown", onDown);
    divider.addEventListener("touchstart", onDown, { passive: false });
    global.addEventListener("mousemove", onMove);
    global.addEventListener("touchmove", onMove, { passive: false });
    global.addEventListener("mouseup", onUp);
    global.addEventListener("touchend", onUp);

    // Expose the control so a test (or a caller) can read/set the split without a pointer.
    const handle = {
      setFraction(f) {
        fraction = f;
        place();
      },
      getFraction() {
        return fraction;
      },
      divider: divider,
    };
    (DE.swipes = DE.swipes || {})[wrapId] = handle;
    return handle;
  };

  // WB-20 — a small overview map (`miniId`) that follows the main map (`mainId`): it recentres on
  // every main move at a lower zoom and draws a rectangle of the main map's current view, so a
  // viewer sees where the main map sits in the wider area. Clicking the overview recentres the main
  // map. The overview's own pan/zoom are disabled — it is a locator, not a second controllable map.
  DE.minimap = function (mainId, miniId, options) {
    const main = DE.maps[mainId];
    const mini = DE.maps[miniId];
    if (!main || !mini) {
      throw new Error("de_maplibre: minimap needs the main and overview maps built");
    }
    options = options || {};
    const zoomOffset = options.zoomOffset != null ? options.zoomOffset : 4;
    for (const handler of [
      "boxZoom",
      "scrollZoom",
      "dragPan",
      "dragRotate",
      "doubleClickZoom",
      "touchZoomRotate",
      "keyboard",
    ]) {
      if (mini[handler] && mini[handler].disable) mini[handler].disable();
    }

    function viewRect() {
      const b = main.getBounds();
      const w = b.getWest();
      const s = b.getSouth();
      const e = b.getEast();
      const n = b.getNorth();
      return {
        type: "Feature",
        properties: {},
        geometry: {
          type: "Polygon",
          coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
        },
      };
    }

    const RECT = "de-view-rect";
    function sync() {
      mini.jumpTo({
        center: main.getCenter(),
        zoom: Math.max(0, main.getZoom() - zoomOffset),
      });
      if (!mini.getSource(RECT)) {
        mini.addSource(RECT, { type: "geojson", data: viewRect() });
        mini.addLayer({
          id: RECT,
          type: "line",
          source: RECT,
          paint: { "line-color": "#ee5555", "line-width": 2 },
        });
      } else {
        mini.getSource(RECT).setData(viewRect());
      }
    }

    main.on("move", sync);
    mini.on("click", (ev) => main.easeTo({ center: ev.lngLat }));
    sync();

    const handle = { sync: sync };
    (DE.minimaps = DE.minimaps || {})[miniId] = handle;
    return handle;
  };
})(typeof window !== "undefined" ? window : this);
