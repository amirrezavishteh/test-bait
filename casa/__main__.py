"""Enable ``python -m casa ...`` as an alias for the CLI."""

from __future__ import annotations

import sys

from casa.cli import main

if __name__ == "__main__":
    sys.exit(main())
