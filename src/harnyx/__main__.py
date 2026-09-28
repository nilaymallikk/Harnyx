"""Enable ``python -m harnyx`` as an alias for the ``harnyx`` console script."""

from __future__ import annotations

from harnyx.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
