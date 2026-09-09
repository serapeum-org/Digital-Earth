"""Enable ``python -m digitalearth`` as an alias for the ``digitalearth`` console script (RP.11)."""
from digitalearth.ops.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
