"""Entry point for `GoldLive.exe` when a user double-clicks it.

Opens the control panel rather than printing CLI usage, because the person who
extracted a ZIP wants a START button, not a subcommand list. The full CLI
remains available as `GoldLive.exe <command>`.
"""

from __future__ import annotations

import sys

from runtime.cli import main

if __name__ == "__main__":
    # No arguments means a double-click, which should open the control panel.
    sys.exit(main(sys.argv[1:] or ["panel"]))
