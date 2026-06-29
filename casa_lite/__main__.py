"""Enable ``python -m casa_lite ...`` as an alias for the CLI."""

from __future__ import annotations

import sys

from casa_lite.cli import main

if __name__ == "__main__":
    sys.exit(main())
