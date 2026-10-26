"""Sphinx configuration for the StillFace documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "StillFace"
copyright = "2026, StillFace contributors"
author = "StillFace contributors"
release = "latest"

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# ── sphinx-autoapi ─────────────────────────────────────────────────────────────

autoapi_dirs = ["../stillface"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_add_toctree_entry = True
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"


def skip_undocumented_attributes(app, what, name, obj, skip, options):
    """Hide bare attributes that have no docstring from the API docs.

    Args:
        app: The Sphinx application object.
        what: The type of the object, for example ``"attribute"``.
        name: The fully-qualified name of the object.
        obj: The object itself.
        skip: Whether the object would be skipped by default.
        options: The options given to the directive.

    Returns:
        True if the object should be skipped.
    """
    if what == "attribute" and not obj.docstring:
        return True
    return skip


def setup(app):
    """Connect the skip handler to autoapi."""
    app.connect("autoapi-skip-member", skip_undocumented_attributes)


# ── Napoleon (Google docstrings) ───────────────────────────────────────────────

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# ── MyST ───────────────────────────────────────────────────────────────────────

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# ── Furo theme ─────────────────────────────────────────────────────────────────

html_theme = "furo"
html_static_path = ["assets"]
html_logo = "assets/logo.svg"
html_baseurl = "https://fodorad.github.io/StillFace/"
exclude_patterns = ["_build"]
