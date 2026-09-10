#!/usr/bin/env python3
"""Backward-compatible entry point; prefer the installed `kbmem-search` command."""

from kbmem.cli import main


if __name__ == "__main__":
    main()
