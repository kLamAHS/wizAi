"""Running a Deimos bot script alongside the combat policy.

Teleporting to the quest marker only gets you as far as the marker. The
things people actually farm — a specific mob, a dungeon loop, a sigil —
are written as **deimoslang** scripts, and Deimos ships a full compiler
and VM for them (`Deimos/src/deimoslang/`).

Rather than invent a second scripting language, this runs those. Deimos's
own runner is (`Deimos.py:2137-2152`)

    while True:
        v = VM(walker.clients)          # EVERY client
        v.load_from_text(source)
        v.running = True
        while v.running:                # a tight loop
            await v.step()
        if v.killed:
            break
        await asyncio.sleep(1)          # and it starts again

and three details of that are load-bearing, all three of which this
module got wrong the first time.

**Every client, one VM.** deimoslang addresses wizards as `p1`..`p4`,
and `VM.player_by_num` returns `None` for an index it does not have
(`vm.py:135-141`) rather than raising -- so a one-client VM does not
fail loudly on `p2`, it fails as `AttributeError: 'NoneType'` somewhere
far away. Building one runner per seat, each over its own client, is
worse still: four copies of the same quester each believing it is `p1`,
walking four wizards to four places. A script is a *party* thing, so
there is one VM over all the clients.

**A tight loop, not one step per tick.** `VM.step()` executes exactly
one instruction (`vm.py:1613`). A real quester is not small -- the
TTS Arc 1 script people actually use is 14,427 lines and compiles to
18,366 instructions -- so at one instruction per half-second service
tick it needs two and a half hours to reach the end of the program
once, and its opening `Close_Menus` block alone takes seventeen
seconds to do what Deimos does instantly. That is not slow, it is
indistinguishable from broken, and it is what "the script does nothing"
means. `ScriptRunner.run_for` steps in a time-boxed burst instead.

**It restarts.** A deimoslang program that runs off the end is meant to
be reloaded and run again; scripts are written assuming it (`# WARNING:
If the script ever restarts...`). Only `kill` ends it.

**Combat is the one thing a script must not do.** A script that fights
would be racing wizAi's policy for the same cards, so the burst runs
only while the wizard is out of combat and only while holding that
wizard's drive lock -- a `waitfor combat` line then parks harmlessly
until the policy has finished the duel.

**Expert mode only.** Deimos has a second, older format -- a flat list
of one-line commands, run through `parse_command` with a preprocessing
pass for `webpage`/`pull`/`embed` (`Deimos.py:2153-2168`). Those are not
deimoslang and do not compile; `expert_mode` says so rather than
letting the compiler produce a baffling parse error on line 1.
"""
import os

from .deimos_path import DEIMOS_ROOT, ensure_path as _ensure_path, install_hint

#: What Deimos's own GUI keys the expert-mode path on (`Deimos.py:2137`).
EXPERT_HEADER = "###deimos_expertmode"


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


def is_expert(source: str) -> bool:
    """Is this deimoslang, or Deimos's older one-command-per-line format?

    The same test Deimos's own GUI makes (`Deimos.py:2137`). The two are
    different languages sharing a text box: only the expert one goes
    through the compiler.
    """
    return (source or "").lstrip().startswith(EXPERT_HEADER)


def wants_clients(source: str) -> int:
    """How many wizards the script's own header says it needs, or 0.

    Scripts carry `# @clients: > 1` / `# @clients: 4` metadata. It is a
    comment as far as the compiler is concerned, but it is the author
    saying what the program assumes -- and a four-wizard quester run
    with one client hooked does not fail, it walks one wizard into a
    dungeon and waits forever for three that are not there.
    """
    import re

    m = re.search(r"^#\s*@clients:\s*(>=?|=)?\s*(\d+)", source or "",
                  re.MULTILINE | re.IGNORECASE)
    if not m:
        return 0
    n = int(m.group(2))
    return n + 1 if m.group(1) == ">" else n


def mentions_clients(source: str) -> int:
    """The highest wizard the text refers to at all, or 0.

    An upper bound, not a requirement: TTS Arc 1 declares
    `@clients: > 1` and still names p4 801 times, behind its own
    configuration flags. So this cannot refuse a script -- it would
    refuse every party script ever written for four and run with two.
    It is worth saying out loud, though, because the parts it guards
    are not always guarded: `teleport client 3` with two wizards
    hooked is an AttributeError the VM never advances past.
    """
    import re

    found = [int(m) for m in re.findall(r"\bp([1-9])\b", source or "")]
    return max(found) if found else 0


def check(source: str):
    """(ok, reason) — does this script compile?

    Worth doing before a run rather than after: a typo on line 40 of a
    pasted script should be an error message in the window, not a
    traceback twenty seconds into a fight.
    """
    ok, reason = available()
    if not ok:
        return False, reason
    if not is_expert(source):
        # Named rather than compiled anyway. Deimos's simple format is
        # not deimoslang, so the compiler reports whatever its first
        # line happens to look like -- a parse error about a token the
        # author never thought of as a token.
        return False, (
            f"this is not an expert-mode script — it has no "
            f"'{EXPERT_HEADER}' first line, so it is Deimos's older "
            f"one-command-per-line format, which is a different language "
            f"and is not supported here. Open it in Deimos, or use an "
            f"expert-mode script.")
    _ensure_path()
    from src.deimoslang import vm

    try:
        probe = vm.VM([])
        probe.load_from_text(source)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    note = f"compiled — {len(probe.program):,} instructions"
    need = wants_clients(source)
    if need > 1:
        note += f", and it says it needs {need} wizards"
    return True, note


def build_vm(clients, source: str):
    """A loaded, running `VM` over `clients`.

    One function because a restart builds one the same way a first start
    does -- Deimos constructs a fresh VM per pass and so does this. See
    `ScriptRunner.restart` for why reloading in place does not work.
    """
    _ensure_path()
    from src.deimoslang import vm

    machine = vm.VM(list(clients))
    machine.load_from_text(source)
    machine.running = True
    return machine


def on_waitfor_timeout(hook):
    """Report a `waitfor` that gave up. Returns (ok, reason).

    The VM's `waitfor` used to poll forever -- see `WAITFOR_TIMEOUTS` in
    `Deimos/src/deimoslang/vm.py`, and `tests/test_deimos_patches.py`
    for why that is patched rather than worked around. A bounded wait
    that gives up silently is only half a fix: the script carries on
    into a retry loop and the run looks normal, so the thing that
    actually went wrong has to be said out loud.

    Reported rather than raised if the symbol is missing, because a
    Deimos update that renames it must not stop wizAi starting. The
    guard test is what catches it properly.
    """
    _ensure_path()
    try:
        from src.deimoslang import vm
    except Exception as exc:
        return False, f"deimoslang did not import ({type(exc).__name__}: {exc})"
    if not hasattr(vm, "on_waitfor_timeout"):
        return False, ("this Deimos has no `on_waitfor_timeout` hook, so a "
                       "`waitfor` that gives up will do it silently — the "
                       "timeout patch in vm.py has probably been lost to an "
                       "update (see tests/test_deimos_patches.py)")
    vm.on_waitfor_timeout = hook
    return True, ""


def on_party_task_failed(hook):
    """Report a wizard that dropped out of a mass instruction. (ok, reason).

    Upstream every mass instruction fans out inside an
    `asyncio.TaskGroup`, which cancels every sibling the moment one task
    raises -- so one wizard's transient memory-read failure cancels the
    other three mid-instruction and the whole VM step dies with an
    ExceptionGroup. `PartyTaskGroup` in `Deimos/src/deimoslang/vm.py`
    lets them all finish and collects the failures instead; this is
    where they surface. See `tests/test_deimos_patches.py`.
    """
    _ensure_path()
    try:
        from src.deimoslang import vm
    except Exception as exc:
        return False, f"deimoslang did not import ({type(exc).__name__}: {exc})"
    if not hasattr(vm, "on_party_task_failed"):
        return False, ("this Deimos has no `on_party_task_failed` hook, so a "
                       "wizard that drops out of a mass instruction will do "
                       "it silently — and it may be taking the rest of the "
                       "party's instruction down with it (see "
                       "tests/test_deimos_patches.py)")
    vm.on_party_task_failed = hook
    return True, ""


def on_teleport_result(hook):
    """Report every `navmap_tp`. Returns (ok, reason).

    Upstream that function's returns are all bare -- success, failure
    and never-attempted are one answer -- which is why deimoslang wraps
    `tp` in unbounded retry loops and why a teleport that silently did
    not land is invisible to the script issuing the next instruction.
    See `tests/test_deimos_patches.py`.
    """
    _ensure_path()
    try:
        from src import teleport_math
    except Exception as exc:
        return False, (f"Deimos's teleport_math did not import "
                       f"({type(exc).__name__}: {exc})")
    if not hasattr(teleport_math, "on_teleport_result"):
        return False, ("this Deimos has no `on_teleport_result` hook, so a "
                       "teleport that does not land will be invisible — the "
                       "patch in teleport_math.py has probably been lost to "
                       "an update (see tests/test_deimos_patches.py)")
    teleport_math.on_teleport_result = hook
    return True, ""


def on_teleport_note(hook):
    """Report a teleport that overrode the loading gate. Returns (ok, reason).

    Its own channel rather than a third kind of result, because it is
    neither: the teleport was attempted, and whether it landed comes
    through `on_teleport_result` as usual. What this says is that
    `navmap_tp` decided the client's "loading" flag was stale and went
    ahead anyway -- a judgement call, and one a human watching a run
    should be able to see us make.
    """
    _ensure_path()
    try:
        from src import teleport_math
    except Exception as exc:
        return False, (f"Deimos's teleport_math did not import "
                       f"({type(exc).__name__}: {exc})")
    if not hasattr(teleport_math, "on_teleport_note"):
        return False, ("this Deimos has no `on_teleport_note` hook, so a "
                       "teleport that pushed through a stale loading flag "
                       "will not be visible — the patch in teleport_math.py "
                       "has probably been lost to an update (see "
                       "tests/test_deimos_patches.py)")
    teleport_math.on_teleport_note = hook
    return True, ""


class ScriptRunner:
    """One deimoslang program, run in time-boxed bursts."""

    #: seconds of instructions per burst. The service task holds this
    #: wizard's drive lock for the whole burst, so this is also the
    #: longest a hotkey press can be made to wait behind the script --
    #: which is why it is a fraction of a second and not the tight loop
    #: Deimos runs. At a few thousand instructions a second it is still
    #: three orders of magnitude more program than one step per tick.
    SLICE = 0.5
    #: instructions in a burst, whatever the clock says. A `sleep 0`
    #: loop would otherwise spin the whole slice with the wheel held.
    MAX_STEPS = 20000
    #: how long ONE instruction may block before the VM is written off.
    #:
    #: Generous, because the instructions that block legitimately block
    #: for a long time: `waitforzonechange completion` waits out a
    #: loading screen and the TTS Arc 1 script has 122 of them. But
    #: bounded, because there is no timeout inside the VM at all -- a
    #: zone change that never comes waits for the rest of the run,
    #: holding this wizard's wheel.
    #:
    #: A step that has to be cancelled is not survivable: cancelling
    #: mid-instruction leaves the VM's own task state half-applied, so
    #: the runner is marked for a rebuild rather than stepped again.
    STEP_LIMIT = 180.0

    #: identical failures at the same instruction before the VM is
    #: written off and reloaded.
    #:
    #: deimoslang has instructions that raise without advancing the
    #: instruction pointer, and the VM has no handler for them. The
    #: reachable one here is `teleport client N` for a wizard the party
    #: does not have: `player_by_num` answers None (`vm.py:137`) and
    #: `target_client.body` is an AttributeError (`vm.py:1414-1419`),
    #: thrown before the arm's `ip += 1` (`vm.py:1826`). Every later
    #: burst re-enters the same instruction and throws the same way,
    #: for the rest of the run.
    #:
    #: The `@clients` header cannot catch it. TTS Arc 1 declares
    #: `@clients: > 1` and still contains `teleport client 3` and
    #: `teleport client 4` behind its own configuration flags -- 18
    #: reachable sites with two wizards hooked, 7 with three. So the
    #: recovery is here rather than in a pre-flight: a script that is
    #: stuck on one instruction is restarted, which is what Deimos does
    #: with a program that fails for any other reason.
    STUCK_AT = 25

    def __init__(self, machine, source, clients=()):
        self.vm = machine
        self.source = source
        #: what the VM was built over, so a restart can rebuild it
        self.clients = list(clients)
        self.failures = 0
        self.last_error = ""
        self.finished = False
        #: instructions executed since the runner was made, across
        #: restarts. The only honest answer to "is the script doing
        #: anything", and the reason the old design's failure was
        #: invisible: one per half-second looks identical to none.
        self.steps = 0
        self.restarts = 0
        #: set when the VM cannot be trusted to carry on -- an
        #: instruction was cancelled part-way, or the program is wedged
        #: on one that throws without advancing. See `STUCK_AT`.
        self.stale = False
        self._stuck_ip = None
        self._stuck = 0

    @property
    def running(self):
        return bool(getattr(self.vm, "running", False))

    @property
    def killed(self):
        return bool(getattr(self.vm, "killed", False))

    async def step(self) -> bool:
        """Advance one instruction. False if it raised or finished."""
        import asyncio

        if not self.running:
            self.finished = True
            return False
        try:
            await asyncio.wait_for(self.vm.step(), self.STEP_LIMIT)
            self.steps += 1
            self.failures = 0
            return True
        except asyncio.TimeoutError:
            self.failures += 1
            self.stale = True
            self.last_error = (
                f"one instruction ran for {self.STEP_LIMIT:.0f}s without "
                f"finishing — a 'waitfor…' whose condition never came. "
                f"Cancelling it leaves the VM half-way through an "
                f"instruction, so the script is being reloaded")
            return False
        except Exception as exc:
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            here = self._ip()
            if here is not None and here == self._stuck_ip:
                self._stuck += 1
            else:
                self._stuck_ip, self._stuck = here, 1
            if self._stuck >= self.STUCK_AT:
                self.stale = True
                self.last_error = (
                    f"stuck on one instruction — it has raised "
                    f"{self._stuck} times without the script moving on "
                    f"({self.last_error}). Reloading, because an "
                    f"instruction that throws before advancing is "
                    f"re-entered for the rest of the run otherwise")
            return False

    def _ip(self):
        """Where the VM is, or None if it will not say."""
        task = getattr(self.vm, "current_task", None)
        return getattr(task, "ip", None)

    async def run_for(self, seconds=None, should_stop=None) -> int:
        """Step until the budget runs out. Returns instructions executed.

        `should_stop()` is checked between instructions and is how the
        caller keeps its guarantees -- a duel starting mid-burst ends
        the burst rather than leaving the script clicking through the
        policy's planning phase.
        """
        import asyncio
        import time

        budget = self.SLICE if seconds is None else seconds
        deadline = time.monotonic() + budget
        done = 0
        while done < self.MAX_STEPS and time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                break
            if not await self.step():
                break
            done += 1
            # Cooperative: a burst that never yields would block the
            # fight loops of the other three wizards on the same loop.
            if done % 64 == 0:
                await asyncio.sleep(0)
        return done

    def restart(self) -> bool:
        """Run the program again from the top. False if it was killed.

        A program that runs off the end is not finished -- Deimos loops
        it (`Deimos.py:2144-2152`) and questers are written expecting
        that. Only `kill` ends a run.

        A **fresh VM**, which is what Deimos builds each pass, and not
        `load_from_text` on the existing one. `load_from_text` assigns
        `program` and nothing else (`vm.py:127-129`): `current_task.ip`
        is still past the end of the old program and `current_task.
        running` is still the False the epilogue set when it got there
        (`vm.py:1833-1834`), so the very next `step()` takes the
        `if not self.current_task.running` branch (`vm.py:1619-1621`)
        and returns without executing anything -- for ever. `until`
        state, timers and constants would survive too. Rebuilding costs
        one compile per pass and leaves nothing behind.
        """
        if self.killed:
            return False
        try:
            machine = build_vm(self.clients, self.source)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        self.vm = machine
        self.restarts += 1
        self.finished = False
        self.stale = False
        self.failures = 0
        self._stuck_ip, self._stuck = None, 0
        return True

    def stop(self):
        try:
            self.vm.kill()
        except Exception:
            pass
        self.finished = True


def make_runner(clients, source: str):
    """Compile `source` against every client, or raise with a reason.

    `clients` is the whole party. deimoslang scripts address wizards as
    `p1`..`p4` and a VM built over one client answers `None` for the
    rest (`vm.py:135-141`), which surfaces as an `AttributeError` deep
    inside an instruction handler rather than as "this script needs more
    wizards than you gave it". A single client is still accepted -- most
    scripts are solo -- but it is passed as a party of one, not as the
    only thing that exists.
    """
    ok, reason = available()
    if not ok:
        raise RuntimeError(reason)
    if not is_expert(source):
        raise RuntimeError(check(source)[1])

    # Both of these before the VM import, because both are complaints
    # about the script rather than about the machine it would run on.
    party = list(clients) if isinstance(clients, (list, tuple)) else [clients]
    need = wants_clients(source)
    if need > len(party):
        raise RuntimeError(
            f"this script says it needs {need} wizards (its '@clients' "
            f"header) and {len(party)} {'is' if len(party) == 1 else 'are'} "
            f"hooked. Running it anyway would not merely skip the "
            f"p{len(party) + 1}… parts: the VM answers None for a wizard it "
            f"does not have, commands over it become silent no-ops, and "
            f"CONDITIONS over it come back true from an empty loop — so a "
            f"'while p{len(party) + 1} …' whose body does nothing never "
            f"ends. Set 'wizards' to {need} and log them all in.")

    return ScriptRunner(build_vm(party, source), source, party)
