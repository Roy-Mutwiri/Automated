"""Entry point for `GoldLive Setup.exe`.

A separate executable purely so the ZIP has one obvious thing to double-click.
It runs provisioning and stops; it never starts a session.
"""

from __future__ import annotations

import sys

from runtime.setup_wizard import main

if __name__ == "__main__":
    sys.exit(main())
