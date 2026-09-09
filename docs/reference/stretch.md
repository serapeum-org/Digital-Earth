# Composite contrast stretch

A composite turns three raster bands into one image by stretching each channel into `[0, 1]`. Every backend
that draws one needs the same three pieces — the per-channel bounds, the stretch itself, and the argument
checks around them — so they live in `digitalearth.base.stretch` and both the matplotlib and HoloViz tiers
consume them. Keeping one copy is what stops the tiers drifting; they had already grown two stretches that
disagreed about an all-nodata channel.

The bounds are ordinary numbers, which is what makes a stable time-lapse possible. Derived per call, each
frame gets its own black and white point and the clip pumps as the scene brightens and darkens; derive them
**once** over the whole stack and hand the same pair to every frame, and the sequence holds one stretch.
`Map.animate` does exactly that for `kind="rgb_composite"` / `"hsv_composite"`, and `InteractiveMap.rgb`
accepts the same `limits` so a HoloViz sequence can be held the same way.

A channel with no finite cell reports `(nan, nan)` rather than raising: that means "no bound for this
channel", and `stretch_to_unit` answers it from the frame in hand instead of inventing a span.

::: digitalearth.base.stretch.channel_limits

::: digitalearth.base.stretch.stretch_to_unit

::: digitalearth.base.stretch.require_three_bands

## `ChannelLimits`

The type the bounds travel as: a sequence of one `(lo, hi)` pair per channel, in channel order. A
non-finite pair means *no bound for this channel* — `stretch_to_unit` answers that from the frame in
hand rather than inventing a span.

## `DEFAULT_COMPOSITE_BANDS`

The `(1, 2, 3)` a composite maps to its channels when the caller names none. Defined once here so the
renderers and the animation scan cannot drift apart about what a bare composite means.

