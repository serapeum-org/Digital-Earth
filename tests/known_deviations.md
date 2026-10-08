<!-- Generated from the conformance and renderer suites' drift tables; do not edit by hand. Run
`python -m tests.test_known_deviations` to regenerate it; tests/test_known_deviations.py fails if it
drifts. -->

# Known cross-tier deviations

The accepted ways a backend legitimately differs from the others, consolidated from the four drift
tables the conformance and renderer suites enforce (`NO_PORTABLE_CHANNELS`, `CANNOT_HIDE_AT_BUILD`,
`NO_PORTABLE_CLASSIFICATION` and `UNDRAWN_KINDS`). Each is both-direction-guarded in its own suite, so
a deviation cannot outlive the defect it excuses; this is the single reviewable list of all of them.

## No portable style channels

The tier records a layer's style in its engine's own spelling rather than folding it into the declared channels, so a
styled layer is fully described only for the tier that drew it.

- **3d** — the builders record their resolved keywords flat in props — size, cmap, scheme — with no fold onto
  declared channels
- **matplotlib** — the caller's keywords stay in props['opts'] and go to cleopatra as they are; nothing on this
  tier folds them into declared channels

## No portable classification

The tier publishes no class edges for a classified layer, so a figure carries none and a legend cannot be reproduced
from the description alone.

- **3d** — this tier has no choropleth() to classify with — `contract.PENDING['3d']` calls a classified fill
  unscheduled, following polygons — so there are no class edges for a figure to carry. Its value-driven layers
  do publish Encoding.by_field('color', …) with a Scale since order 24, but that scale carries vmin/vmax and
  breaks=(), because nothing cut any

## Kinds kept but not drawn

The tier's renderer keeps a layer of this kind but draws nothing from its description — a custom engine object held by
reference rather than rebuilt.

- **interactive · `custom:holoviews`** — a caller's own element, drawn by being kept: `add_layer` holds no
  description to rebuild it from
