from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# app/static: a single page of plain HTML, CSS and JavaScript, with no build step.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def mount_ui(app: FastAPI) -> None:
    """Serves the web UI at /ui/. The page itself needs no key; the API calls it makes do,
    and the page sends the key the user enters."""
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
