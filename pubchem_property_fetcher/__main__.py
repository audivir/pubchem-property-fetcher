"""Command-line entry point for `python -m pubchem_property_fetcher`."""

from __future__ import annotations

if __name__ == "__main__":
    import doctyper

    from pubchem_property_fetcher import main

    app = doctyper.DocTyper()
    app.command()(main)
    app()
