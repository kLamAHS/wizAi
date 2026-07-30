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
#: Wisps and potions are here because the automatic versions run only
#: between fights -- there is no way to top up mid-run without one.
DEFAULTS = {"teleport": "F1", "dialogue": "F2",
            "wisps": "F3", "potion": "F4"}

#: What the GUI offers. Deliberately short: every entry here is a key
#: taken away from every other program for the length of the run, so the
#: list is confined to ones a game or an editor is unlikely to want.
#:
#: The numpad names are wizwalker's spelling, not the obvious one. They
#: were listed here as `NUMPAD0`..`NUMPAD3`, which is not what
#: `Keycode` calls them (`constants.py:111` — `Numeric_pad_0`), so
#: `resolve` returned None and four of the fourteen choices bound
#: nothing at all: pick one and the action is quietly dropped.
KEY_CHOICES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
               "F9", "F10", "Numeric_pad_0", "Numeric_pad_1",
               "Numeric_pad_2", "Numeric_pad_3")


def available():
    """(ok, reason) -- whether hotkeys can be installed at all."""
    try:
        from wizwalker import HotkeyListener, Keycode      # noqa: F401
    except Exception as exc:
        return False, (f"wizwalker's hotkey support did not import ({exc}). "
                       "Global hotkeys need Windows.")
    return True, ""


def resolve(name):
    """A `Keycode` from a name in `KEY_CHOICES`, or None.

    Exact name first. `Keycode` mixes cases -- `F1` but `Numeric_pad_0`
    -- so upper-casing everything can never match half the enum, and a
    name that does not resolve is an action that silently does nothing.
    """
    from wizwalker import Keycode

    name = str(name)
    key = getattr(Keycode, name, None)
    # `is None`, not truthiness: `Keycode` is an IntEnum and a member
    # whose value is 0 is falsy while being perfectly valid.
    return key if key is not None else getattr(Keycode, name.upper(), None)


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
        #: actions whose last press was refused, so a held key reports
        #: "already running" once rather than once per repeat
        self._dropped = set()

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
                await self.listener.add_hotkey(key, self._make(action),
                                               **self._modifiers())
            except Exception as exc:
                # wizwalker raises ValueError("already registered") for a
                # genuine collision, which is misleading -- it is another
                # *program* holding the key, not us. But that message was
                # printed for **every** exception, so a wizwalker API
                # mismatch or a permissions problem sent the user off
                # cycling through all fourteen keys, none of which was
                # ever the problem. Name what actually happened.
                if isinstance(exc, ValueError) and "registered" in str(exc):
                    self.on_status(
                        f"hotkey {key_name} unavailable — another program "
                        f"has it. Pick a different key.")
                else:
                    self.on_status(
                        f"hotkey {key_name} could not be installed — "
                        f"{type(exc).__name__}: {exc}")
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

    @staticmethod
    def _modifiers():
        """`NOREPEAT`, if this wizwalker has it.

        Without it Windows auto-repeats `WM_HOTKEY` for as long as the
        key is held, so leaning on the wisps key fires a stream of
        requests. The queue's dedupe only covers the window in which an
        action sits *waiting* -- a sweep that is already running is not
        in the queue -- so the repeats chained back-to-back sweeps.
        wizwalker strips NOREPEAT before keying the callback
        (`hotkey.py:369`), so dispatch is unaffected.
        """
        try:
            from wizwalker import ModifierKeys
        except Exception:
            try:
                from wizwalker.hotkey import ModifierKeys
            except Exception:
                return {}
        return {"modifiers": ModifierKeys.NOREPEAT}

    def _make(self, action):
        async def fire():
            try:
                accepted = self.on_action(action)
            except Exception:
                return    # a bad callback must not kill the listener
            if accepted is False:
                # Said once per run of drops, not per repeat: a key held
                # down would otherwise fill the status bar with this.
                if action not in self._dropped:
                    self._dropped.add(action)
                    self.on_status(f"{action} is already running — "
                                   f"ignoring the repeat")
            else:
                self._dropped.discard(action)
        return fire
