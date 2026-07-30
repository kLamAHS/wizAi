"""Make a crash reportable instead of a window that just disappears.

Qt is unusually good at losing errors. An unhandled Python exception
inside a virtual method -- `paintEvent`, `resizeEvent`, a slot -- does
not propagate anywhere useful: **PyQt6 calls `qFatal` and the process
aborts**. On Windows, launched from a shortcut, that is a window
vanishing with no message and nothing written down.

Which turns "the Learning tab crashes" into a report nobody can act on,
including the person who wrote it.

Two things fix that, and both are cheap:

  * a `sys.excepthook` that writes the traceback to a file beside the
    repository *and* shows it, so there is something to paste;
  * `faulthandler`, which catches the aborts a Python-level hook cannot
    see -- the `qFatal` path included.

Neither prevents a crash. They make it possible to fix one.
"""
import datetime
import faulthandler
import os
import sys
import traceback

#: written next to the repository, not into a temp directory -- a log
#: nobody can find is a log nobody will send.
LOG_NAME = "wizAi-crash.log"


def log_path():
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(root, LOG_NAME)


def _write(text):
    try:
        with open(log_path(), "a", encoding="utf-8") as fh:
            fh.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
                     f"=====\n{text}\n")
    except Exception:
        pass          # a failure to log must not become the crash


def install(show=None):
    """Route unhandled errors to the log, and to `show(text)` if given.

    `show` is a callable rather than a QMessageBox import, so this module
    stays importable without Qt and can be tested without a display.
    """
    faulthandler.enable(all_threads=True)
    try:
        # Catches the abort path a Python-level hook cannot: PyQt6's
        # qFatal on an exception inside a virtual, which is what a broken
        # paintEvent actually does.
        fh = open(log_path(), "a", encoding="utf-8")
        faulthandler.enable(file=fh, all_threads=True)
    except Exception:
        pass

    previous = sys.excepthook

    def hook(kind, value, tb):
        text = "".join(traceback.format_exception(kind, value, tb))
        _write(text)
        sys.stderr.write(text)
        if show is not None:
            try:
                show(text)
            except Exception:
                pass
        previous(kind, value, tb)

    sys.excepthook = hook

    # Qt installs its own handler for exceptions raised in slots on some
    # builds; threading errors bypass sys.excepthook entirely.
    try:
        import threading

        def thread_hook(args):
            hook(args.exc_type, args.exc_value, args.exc_traceback)

        threading.excepthook = thread_hook
    except Exception:
        pass
    return log_path()
