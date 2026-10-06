"""Engine-free access to each tier's declared ``Capabilities``, shared by the U-5 ratchets.

Both the API snapshot (``tests/test_api_snapshot.py``) and the support matrix (``tests/test_support_matrix.py``)
read the four tiers' capabilities. Each tier's ``capabilities`` module imports without MapLibre, HoloViz or
PyVista — only the plain :class:`~digitalearth.base.capabilities.Capabilities` value — so both ratchets run in
the lean ``dev`` matrix. This holds the one map from a backend name to its capabilities module, and the reader
over it, so neither ratchet repeats it.
"""

import importlib
from typing import Any

#: Backend name → the module holding that tier's ``CAPABILITIES`` value. All four import engine-free.
TIER_CAPABILITY_MODULES = {
    "matplotlib": "digitalearth.static.capabilities",
    "web": "digitalearth.web.capabilities",
    "interactive": "digitalearth.interactive.capabilities",
    "3d": "digitalearth.three_d.capabilities",
}


def tier_capabilities(backend: str) -> dict[str, Any]:
    """Return one tier's declared capabilities as a plain dict.

    Args:
        backend: A key of :data:`TIER_CAPABILITY_MODULES`.

    Returns:
        The tier's :meth:`~digitalearth.base.capabilities.Capabilities.to_dict`.
    """
    module = importlib.import_module(TIER_CAPABILITY_MODULES[backend])
    return module.CAPABILITIES.to_dict()
