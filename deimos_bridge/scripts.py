"""Running a Deimos bot script alongside the combat policy.

Teleporting to the quest marker only gets you as far as the marker. The
things people actually farm — a specific mob, a dungeon loop, a sigil —
are written as **deimoslang** scripts, and Deimos ships a full compiler
and VM for them (`Deimos/src/deimoslang/`).

Rather than invent a second scripting language, this runs those. The VM
is already step-based, which is what makes it composable:

    v = VM([client])
    v.load_from_text(source)
    v.running = True
    while v.running:
        await v.step()

Deimos's own runner is that loop verbatim (`Deimos.py:2142-2149`). Taking
one `step()` per service tick instead of running the loop keeps the live
worker in charge of the fight, exactly as `DeimosQuester` does.

**Combat is the one thing a script must not do.** A script that fights
would be racing wizAi's policy for the same cards. `ScriptRunner` steps
only while the wizard is out of combat, so a `waitfor combat` line parks
harmlessly until the policy has finished the duel.
"""
import os

from .deimos_path import DEIMOS_ROOT, ensure_path as _ensure_path, install_hint


def available():
    """(ok, reason). Is deimoslang importable?"""
    if not os.path.isdir(DEIMOS_ROOT):
        return False, f"no Deimos directory at {DEIMOS_ROOT}"
    _ensure_path()
    try:
        from src.deimoslang import vm  # noqa: F401
        return True, ""
    except Exception as exc:
        # Name the one thing that is missing. The old message printed a
        # fixed list headed by `wizsprinter`, which is not on PyPI at
        # all -- it is vendored at Deimos/libs/wizsprinter and is now put
        # on the path by `ensure_path`, so nobody should be told to
        # install it.
        hint = install_hint(exc)
        return False, (
            f"deimoslang is not importable ({type(exc).__name__}: {exc})."
            + (f"\n\n{hint}" if hint else ""))


def check(source: str):
    """(ok, reason) — does this script compile?

    Worth doing before a run rather than after: a typo on line 40 of a
    pasted script should be an error message in the window, not a
    traceback twenty seconds into a fight.
    """
    ok, reason = available()
    if not ok:
        return False, reason
    _ensure_path()
    from src.deimoslang import vm

    try:
        probe = vm.VM([])
        probe.load_from_text(source)
        return True, f"compiled — {len(probe.program)} instructions"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


class ScriptRunner:
    """One deimoslang program, stepped."""

    def __init__(self, machine, source):
        self.vm = machine
        self.source = source
        self.failures = 0
        self.last_error = ""
        self.finished = False

    @property
    def running(self):
        return bool(getattr(self.vm, "running", False))

    async def step(self) -> bool:
        """Advance one instruction. False if it raised or finished."""
        if not self.running:
            self.finished = True
            return False
        try:
            await self.vm.step()
            self.failures = 0
            return True
        except Exception as exc:
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def stop(self):
        try:
            self.vm.kill()
        except Exception:
            pass
        self.finished = True


def make_runner(client, source: str):
    """Compile `source` against `client`, or raise with a readable reason."""
    ok, reason = available()
    if not ok:
        raise RuntimeError(reason)
    _ensure_path()
    from src.deimoslang import vm

    machine = vm.VM([client])
    machine.load_from_text(source)
    machine.running = True
    return ScriptRunner(machine, source)
