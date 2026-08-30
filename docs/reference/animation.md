# Saving animations

Render a clip once and derive every other format from the written file.

Drawing frames costs far more than encoding them, so `save_animation` encodes the animation to a video and —
when a GIF is also wanted — reads the frames back off that file rather than redrawing them. It also raises the
intermediate video to full chroma in that case, because a subsampled source permanently caps how much colour
detail can survive the GIF palette.

Reachable as `Map.save_animation` and `TexturedGlobe.save_animation`, or directly as the function below.

::: digitalearth.animation.save_animation
