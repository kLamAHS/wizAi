"""Importing Deimos's own modules out of the subtree, without installing.

`Deimos/` is vendored here as a subtree, and so are the two libraries its
questing and scripting need. Both of those things get in the way of
simply importing `src.questing` or `src.deimoslang`:

  * **`src.*` is only importable with `Deimos/` on the path**, because
    Deimos is an application rather than a package.

  * **`wizwalker.extensions.wizsprinter` is not a PyPI package.** It is a
    workspace member at `Deimos/libs/wizsprinter`, and it installs *into*
    the `wizwalker` namespace rather than beside it. Telling anyone to
    `pip install wizsprinter` is therefore wrong advice -- which is what
    this module exists to stop, because that advice was being printed.

The vendored `wizwalker/extensions/__init__.py` already scans `sys.path`
for `wizwalker/extensions` directories and merges them, so putting
`Deimos/libs/wizsprinter` on the path is enough there. A fork without
that loop needs `__path__` extending by hand, so both are done.

`requires-python = ">=3.13"` in `libs/wizsprinter/pyproject.toml` is
declared metadata, not a real floor -- the sources compile on 3.11, which
is why importing them directly works where `pip install` refuses.
"""
import os
import sys

DEIMOS_ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "Deimos")

#: Vendored libraries that overlay into the `wizwalker` namespace.
_OVERLAYS = ("wizsprinter",)

#: import name -> what to pip install for it. Only the ones that really
#: are separate PyPI packages; wizsprinter deliberately is not here.
PIP_NAMES = {
    "lark": "lark",
    "thefuzz": "thefuzz",
    "loguru": "loguru",
    "yaml": "pyyaml",
    "requests": "requests",
    "pypresence": "pypresence",
    "pyperclip": "pyperclip",
    "cryptography": "cryptography",
    "win32api": "pywin32",
    "win32gui": "pywin32",
}


def ensure_path():
    """Put `Deimos/` and its vendored extensions where imports find them."""
    if os.path.isdir(DEIMOS_ROOT) and DEIMOS_ROOT not in sys.path:
        sys.path.insert(0, DEIMOS_ROOT)
    for name in _OVERLAYS:
        _overlay(name)


def _overlay(name):
    """Make `wizwalker.extensions.<name>` resolve to the vendored copy.

    By extending the already-imported package's `__path__`, and
    deliberately **not** by putting `libs/<name>` on `sys.path`. That
    directory contains a `wizwalker/` with no `__init__.py`, so adding it
    to the path makes `wizwalker` resolvable as a *namespace* package --
    which shadows the real one wherever the real one is missing or found
    later, and turns "wizsprinter is absent" into "wizwalker is broken".
    Extending `__path__` can only ever add a submodule to a wizwalker
    that already imported.
    """
    sub = os.path.join(DEIMOS_ROOT, "libs", name, "wizwalker", "extensions")
    if not os.path.isdir(sub):
        return
    try:
        import wizwalker.extensions as ext
    except Exception:
        return          # no wizwalker here at all; nothing to overlay onto
    if sub not in ext.__path__:
        ext.__path__.append(sub)


def missing_requirement(exc):
    """What to actually install for this import error, or None.

    Named specifically rather than as a fixed shopping list. The old
    message printed the same seven packages whatever went wrong, and
    headed them with `wizsprinter` -- which does not exist on PyPI, so
    the one instruction anybody would copy could not succeed.
    """
    name = getattr(exc, "name", None)
    if not name:
        return None
    top = name.split(".")[0]
    if top == "wizwalker":
        return None                     # ours to fix, not theirs
    return PIP_NAMES.get(top, top)


def install_hint(exc):
    """A line worth showing under an import failure."""
    name = (getattr(exc, "name", "") or "").split(".")[0]
    if name == "wizwalker":
        return ("wizwalker itself is missing, which means this is not the "
                "environment the live tools run in — see RUNNING_LIVE.md. "
                "Nothing needs installing for wizsprinter: it is vendored "
                "at Deimos/libs/wizsprinter.")
    pkg = missing_requirement(exc)
    if not pkg:
        return ""
    return (f"Install it into this venv:\n"
            f"    .venv\\Scripts\\python.exe -m pip install {pkg}")
