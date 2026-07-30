"""System-wide hotkeys, so the game never loses focus.

The GUI's *Teleport to quest* button works, but using it costs a
alt-tab out of a full-screen game and an alt-tab back -- which in
practice means you stop using it. A hotkey is the same action without
leaving the client.

These are **global** hotkeys, not keys sent to the game: wizwalker's
`HotkeyListener` goes through Win32 `RegisterHotKey`, which means

  * they fire no matter which window has focus, and
  * while registered, the key is **taken away from every other program**,
    including Wizard101 itself.

That second point is why the defaults are function keys and why they are
configurable. Bind something the game uses and you will lose it in the
game.

Registration also fails if another running program already owns the
combination. wizwalker reports that as `ValueError("... already
registered")`, which reads as though *we* registered it twice; it is
translated here into something a person can act on.

Windows-only, like everything that touches wizwalker. `available()` says
so without raising, so the GUI can offer the checkbox and explain rather
than crash.
"""

#: action -> default key name. Function keys because Wizard101 binds
#: almost nothing above F1 and they are easy to hit without looking.
DEFAULTS = {"teleport": "F1", "dialogue": "F2"}

#: What the GUI offers. Deliberately short: every entry here is a key
#: taken away from every other program for the length of the run, so the
#: list is confined to ones a game or an editor is unlikely to want.
KEY_CHOICES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
               "F9", "F10", "NUMPAD0", "NUMPAD1", "NUMPAD2", "NUMPAD3")


def available():
    """(ok, reason) -- whether hotkeys can be installed at all."""
    try:
        from wizwalker import HotkeyListener, Keycode      # noqa: F401
    except Exception as exc:
        return False, (f"wizwalker's hotkey support did not import ({exc}). "
                       "Global hotkeys need Windows.")
    return True, ""


def resolve(name):
    """A `Keycode` from a name in `KEY_CHOICES`, or None."""
    from wizwalker import Keycode

    return getattr(Keycode, str(name).upper(), None)


class Hotkeys:
    """Global hotkeys bound to actions, for the length of a live run.

    Owns nothing but the listener. Actions are dispatched by name to a
    callback, so this never touches the client, the GUI, or the fight --
    the worker already has a queue that is safe to append to from
    anywhere, and that is where a keypress lands.
    """

    def __init__(self, bindings, on_action, on_status=None):
        """
        Args:
            bindings: {action name: key name}, e.g. {"teleport": "F1"}.
            on_action: called with the action name when its key is hit.
            on_status: optional, called with a line worth showing.
        """
        self.bindings = dict(bindings or {})
        self.on_action = on_action
        self.on_status = on_status or (lambda _m: None)
        self.listener = None
        #: action -> key name, for the ones that actually registered
        self.installed = {}

    async def start(self):
        """Register every binding. Returns the ones that took.

        A key that will not register is reported and skipped rather than
        raising: losing one hotkey is not a reason to lose the run, and
        the usual cause -- another program already holds it -- is
        something only the person at the keyboard can fix.
        """
        ok, reason = available()
        if not ok:
            self.on_status(reason)
            return {}

        from wizwalker import HotkeyListener

        self.listener = HotkeyListener()
        for action, key_name in self.bindings.items():
            key = resolve(key_name)
            if key is None:
                self.on_status(f"hotkey: no such key {key_name!r}")
                continue
            try:
                await self.listener.add_hotkey(key, self._make(action))
            except Exception:
                # wizwalker raises ValueError("already registered") for
                # this, which is misleading -- the collision is with
                # another *program*, not with us.
                self.on_status(
                    f"hotkey {key_name} unavailable — another program has "
                    f"it. Pick a different key.")
                continue
            self.installed[action] = key_name

        if not self.installed:
            self.listener = None
            return {}

        self.listener.start()
        self.on_status("hotkeys: " + ", ".join(
            f"{k} = {a}" for a, k in self.installed.items()))
        return dict(self.installed)

    async def stop(self):
        if self.listener is None:
            return
        try:
            await self.listener.stop()
        except Exception:
            pass          # tearing down must never fail a run's shutdown
        finally:
            self.listener = None
            self.installed = {}

    def _make(self, action):
        async def fire():
            try:
                self.on_action(action)
            except Exception:
                pass      # a bad callback must not kill the listener
        return fire
