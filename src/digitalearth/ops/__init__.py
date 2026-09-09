"""ops — the operational surface: what drives the package rather than what draws.

These modules are neither a rendering backend nor shared drawing logic, which is why they sat loose at the
package root for so long. They *use* the backends: render a directory of rasters (:mod:`~digitalearth.ops.batch`),
build a static HTML gallery from the results (:mod:`~digitalearth.ops.browser`), expose both from a command line
(:mod:`~digitalearth.ops.cli`), and discover third-party extensions through entry points
(:mod:`~digitalearth.ops.plugins`).

Grouping them here leaves the package root as just the public facade (``__init__``) and the backend dispatcher
(``api``).
"""
