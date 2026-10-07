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
})(typeof window !== "undefined" ? window : this);
