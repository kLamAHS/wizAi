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
import re

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


def debug_state(source: str):
    """True/False as the script's own `var DebugMode` has it, or None.

    The presets ship `var DebugMode = True` on line 10, described there
    as "Will show exact steps more in depth. Useful for determining
    breaks." -- which is exactly the question a stuck run asks.
    """
    import re

    m = re.search(r"^\s*var\s+DebugMode\s*=\s*(True|False)\b", source or "",
                  re.MULTILINE)
    return None if not m else (m.group(1) == "True")


def set_debug(source: str, on: bool) -> str:
    """The script with `var DebugMode` forced on or off.

    Line-count preserving, like `restart_source`, because the stuck
    instruction pointer is reported as a number and a rewrite that
    shifts lines makes every previous export's numbers wrong.
    """
    import re

    want = "True" if on else "False"

    def swap(m):
        return f"{m.group(1)}{want}{m.group(3)}"

    return re.sub(r"^(\s*var\s+DebugMode\s*=\s*)(True|False)(\b.*)$",
                  swap, source or "", count=1, flags=re.MULTILINE)


#: loguru records from the deimoslang VM, and nothing else.
VM_LOGGERS = ("src.deimoslang", "deimoslang")


def capture_prints(on_message):
    """Route the script's `print` output to `on_message`. Returns a
    callable that stops it, or None if it could not be attached.

    deimoslang's `print` compiles to `log_single`/`log_multi`, and both
    end in `logger.debug(...)` -- loguru (`vm.py:30`). loguru's default
    sink is stderr, so on a GUI run the script's own commentary went to
    a console nobody was reading: `DebugMode` has shipped **True** in
    every preset this whole time and not one line of it ever reached an
    export. The scripts' `print`s name the dispatch leg they are on
    ("Broken quest detected. p1 tracking_quest [get some bling]"), which
    is the one thing a wizard stuck in a route cannot otherwise say.
    """
    try:
        from loguru import logger
    except Exception:
        return None

    def only_the_vm(record):
        name = record.get("name") or ""
        return any(name.startswith(p) for p in VM_LOGGERS)

    def sink(message):
        try:
            on_message(message.record["message"])
        except Exception:
            pass

    try:
        handle = logger.add(sink, level="DEBUG", filter=only_the_vm,
                            format="{message}", enqueue=False)
    except Exception:
        return None

    def stop():
        try:
            logger.remove(handle)
        except Exception:
            pass

    return stop


def unconfigured(source: str):
    """[(name, value)] for variables still at the script's own placeholder.

    Rev 8e5a9c75 is 110 minutes long and the last 40 of them are three
    wizards standing in Krokotopia's hub, out of combat, full health,
    while the script executed 106,000 instructions and moved nobody.
    The cause is in the file's first thirty lines::

        var Main_Account = "QuestingAccountName"
        var Questee2     = "QuestingAccountName"
        var Main_Account_School = "SchoolGoesHere"

    Never filled in. The quester guards most of its friend-teleports
    with `if NOT Main_Account = "QuestingAccountName"`, so an
    unconfigured script does not fail -- it silently skips the one
    mechanism it has for putting a party back together. Three of them
    got through the guards anyway and threw `Could not find friend with
    ... name QuestingAccountName`, which is how this became visible at
    all, and only because `PartyTaskGroup` now reports what it catches.

    So: a wizard falls behind, and the script's own regroup is disabled.
    That is the whole failure, and it is checkable before the run starts.

    How a placeholder is recognised, without a hardcoded list of them:
    the script COMPARES the variable against the same literal it was
    assigned. `var X = "Y"` alone is just a value; `var X = "Y"` plus
    `if NOT X = "Y"` elsewhere is the author saying "Y means unset".
    That generalises to any script written the same way, which is all of
    the ones built from this template.

    The NOT matters, and it took filling the schools in correctly to
    find out why. The template dispatches on them::

        if NOT Main_Account_School = "SchoolGoesHere" {
            if Main_Account_School = "Life" {
            } elif Main_Account_School = "Storm" {

    so once `configure` writes "Storm" onto the `var` line -- in the
    script's own spelling, which is the whole point of
    `script_spelling` -- the plain assigned-AND-compared rule sees
    "Storm" assigned and "Storm" compared and calls a correctly
    configured variable a placeholder, forever.

    A placeholder is what the script tests for the ABSENCE of. Anything
    it dispatches ON is vocabulary. The old rule stays as the fallback
    for a variable with no negative guard at all, where a positive
    `if X = "unset"` is the only evidence available.

    ...and one hardcoded fallback after all. The 2026-08 template
    refresh dropped the guards entirely -- the new scripts gate their
    multi-client branches on `playercount` and use the account vars
    directly (`p2 selectfriend $Main_Account`), so nothing ever
    compares a var against its placeholder and the general rule has no
    evidence to read. The template's own placeholders are stable and
    documented on their `var` lines, so a var still ASSIGNED one of
    them is unconfigured even with nothing comparing it. Worse than
    before, in fact: with the guards gone, an unfilled name is no
    longer silently skipped -- `selectfriend "QuestingAccountName"`
    actively runs and fails.
    """
    import re

    found = []
    for name, value in re.findall(r'^\s*var\s+(\w+)\s*=\s*"([^"]*)"',
                                  source or "", re.MULTILINE):
        if not value:
            continue
        pattern = rf'{re.escape(name)}\s*=\s*"{re.escape(value)}"'
        # More than one is the `var` line plus at least one comparison.
        if len(re.findall(pattern, source)) <= 1:
            if value in TEMPLATE_PLACEHOLDERS:
                found.append((name, value))
            continue
        negated = re.findall(rf'\bNOT\s+{re.escape(name)}\s*=\s*"([^"]*)"',
                             source)
        if negated:
            if value in negated:
                found.append((name, value))
            continue
        found.append((name, value))
    return found


#: Where preset scripts live. Kept under `data/` rather than beside this
#: module because `deimos_bridge/scripts.py` already owns that name, and
#: a `scripts/` package next to it would shadow the module on import.
PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "scripts")


def presets():
    """[(title, path)] for the scripts shipped with wizAi, by title.

    Empty is a normal answer -- wizAi ships no scripts of its own, and
    the directory is where an operator's own arc questers go so they are
    a dropdown rather than a paste every session.

    The title is the script's own `# @name:` line if it has one, else
    the filename. Nothing is compiled here: listing has to work even
    when Deimos is not importable, because the dialog that lists them is
    also where you find out Deimos is not importable.
    """
    import glob

    found = []
    for path in sorted(glob.glob(os.path.join(PRESET_DIR, "*.txt"))
                       + glob.glob(os.path.join(PRESET_DIR, "*.deimos"))):
        found.append((preset_title(path), path))
    return sorted(found)


def preset_title(path: str) -> str:
    import re

    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except Exception:
        return os.path.basename(path)
    m = re.search(r"^#\s*@name:\s*(.+)$", head, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return os.path.splitext(os.path.basename(path))[0].replace("_", " ")


def read_preset(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


#: Which of a quester's account variables belongs to which wizard, and
#: which holds a school. The TTS template's own comment settles the
#: order -- "Name of the main account. Also MUST BE SET AS p1" -- and
#: every script built from that template inherits it.
ACCOUNT_VARS = ("Main_Account", "Questee2", "Questee3", "Questee4")
SCHOOL_VARS = ("Main_Account_School", "Questee2_School",
               "Questee3_School", "Questee4_School")

#: The TTS template's own placeholder literals, verbatim. The general
#: assigned-AND-compared rule in `unconfigured` needs no such list --
#: but the 2026-08 template refresh removed every guard comparison, so
#: for those scripts the literals themselves are the only evidence a
#: var was never filled in. See `unconfigured` and `solo_source`.
TEMPLATE_PLACEHOLDERS = ("QuestingAccountName", "SchoolGoesHere")


def unfilled(source: str, party_size=None):
    """Placeholders a party of this size could actually have filled in.

    `Questee4` in a three-wizard party is not an unconfigured setting;
    there is no fourth wizard to put in it, and the script's own guard
    on it is doing exactly what it is for. Reporting it as a defect is
    how a live run at rev 1dcf4193 opened its export with

        the script has 2 setting(s) still at its placeholder value
        (Questee4, Questee4_School)

    which reads as the cause of everything after it and was nothing at
    all. `party_size=None` keeps the whole list, for a pre-flight on a
    script with no party attached yet.
    """
    blanks = unconfigured(source)
    if party_size is None:
        return blanks
    spare = set(ACCOUNT_VARS[int(party_size):])
    spare |= set(SCHOOL_VARS[int(party_size):])
    return [(name, value) for name, value in blanks if name not in spare]


def script_spelling(source: str, var: str, value: str, placeholder: str):
    """`value` spelled the way THIS script spells it, or `value`.

    The schools are the case in point. wizAi holds a seat's school
    lowercase -- it is a dropdown value -- and the TTS template
    dispatches on::

        if NOT Main_Account_School = "SchoolGoesHere" {
            if Main_Account_School = "Life" {
            } elif Main_Account_School = "Storm" {

    Capitalised, and deimoslang's `=` is a plain string comparison. So
    filling in "storm" is the worst of both: the outer guard passes,
    because "storm" is not the placeholder, and every branch inside it
    fails, so the whole dispatch runs and does nothing. Silent, and
    exactly the shape of the bug this module exists to prevent.

    Rather than hardcoding the seven school names, this reads the
    spellings out of the script the same way `unconfigured` reads the
    placeholders: whatever literals the script COMPARES this variable
    against are its vocabulary, and a case-insensitive match against
    that list is the script's own spelling of the value.
    """
    import re

    want = (value or "").strip().lower()
    if not want:
        return value
    for literal in re.findall(rf'{re.escape(var)}\s*=\s*"([^"]*)"', source):
        if literal != placeholder and literal.strip().lower() == want:
            return literal
    return value


#: the keys a deimoslang script presses to advance a dialogue box.
DIALOGUE_KEYS = ("SPACEBAR", "ENTER")


def undeclared_guards(source: str):
    """[(name, line)] for `if X = True` guards on a variable that is never
    declared.

    deimoslang answers an undefined constant with `False`, silently
    (`vm.py:1109`) -- so a guard on a name nobody declared is not a
    syntax error and not a warning, it is a branch that can never be
    taken and a feature that is quietly off.

    Every TTS preset has exactly one, and it is the dialogue clearer::

        if any hasdialogue {
            print "Dialogue detected. Clearing..."
            if Handle_Dialogue = True {
                sameany sendkey SPACEBAR, .1
            }
        }

    `Handle_Dialogue` is used on that line and declared nowhere in
    14,427 lines, so the script announces that it is clearing the box
    and then does not press anything. See `clears_dialogue`.
    """
    import re

    declared = {m.group(1) for m in
                re.finditer(r'^\s*var\s+(\w+)\s*=', source or "", re.M)}
    found = []
    for n, raw in enumerate((source or "").splitlines(), 1):
        line = raw.split("#", 1)[0]
        for m in re.finditer(r'\bif\s+(?:NOT\s+)?(\w+)\s*=\s*(?:True|False)\b',
                             line):
            name = m.group(1)
            if name not in declared and (name, n) not in found:
                found.append((name, n))
    return found


def dead_dialogue_guard(source: str) -> str:
    """The undeclared variable switching this script's dialogue clearer
    off, or "".

    The question wizAi has to answer before deciding whether to run its
    own clicker for a scripted wizard. It used to assume that a running
    script handles its own dialogue, and for every preset it ships that
    is false::

        if any hasdialogue {
            print "Dialogue detected. Clearing..."
            if Handle_Dialogue = True {
                sameany sendkey SPACEBAR, .1
            }
        }

    `Handle_Dialogue` is declared nowhere in 14,427 lines, deimoslang
    answers an undefined constant with `False` (`vm.py:1109`), so the
    press never happens -- the script announces that it is clearing the
    box and then does not press anything. Rev ed709013 caught the
    symptom without the cause: `Dialogue detected. Clearing...` fifteen
    times in eight milliseconds with General Khaba's MORE button
    untouched on all three clients.

    Deliberately narrow. It does NOT ask "does this script ever press
    space" -- the presets do, in per-quest sequences and in the spell
    trainer, and those are not the general handler. It asks whether a
    dialogue press is gated on a name that cannot ever be true, which
    is a fact about one guard and checkable exactly.
    """
    import re

    declared = {m.group(1) for m in
                re.finditer(r'^\s*var\s+(\w+)\s*=', source or "", re.M)}
    keys = "|".join(DIALOGUE_KEYS)
    stack = []
    for raw in (source or "").splitlines():
        line = raw.split("#", 1)[0]
        # Closes first, so `} else {` pops the branch it is leaving
        # before opening the one it enters.
        for _ in range(line.count("}")):
            if stack:
                stack.pop()
        guard = None
        m = re.search(r'\bif\s+(\w+)\s*=\s*(?:True|False)\b', line)
        opens = line.count("{")
        if m and opens:
            guard = m.group(1)
        for i in range(opens):
            # Only the innermost brace this line opens carries the guard.
            stack.append(guard if i == opens - 1 else None)
        if re.search(rf'\bsendkey\s+(?:{keys})\b', line, re.I):
            for name in stack:
                if name is not None and name not in declared:
                    return name
    return ""


def configure(source: str, wizards):
    """(source, filled). Put the party's real names into the script.

    `wizards` is [(name, school)] in seat order, p1 first.

    This is the structural fix for what cost rev 8e5a9c75 forty minutes.
    The quester needs its account names to match the wizards actually
    logged in, an operator has to type them by hand into a 14,000-line
    file, and when they do not the script does not complain -- it guards
    its own friend-teleports with `if NOT Main_Account =
    "QuestingAccountName"` and quietly skips every one of them. The
    party then has no way to regroup and the run ends as a stall.

    wizAi already knows both facts: the name comes off the client's own
    combat member, and the school is what the seat was configured with.
    So it fills them in, and the operator cannot get it wrong.

    Only placeholders are touched -- see `unconfigured`. A name the
    operator typed themselves is theirs, even if it disagrees with what
    wizAi read, because they may be running a script whose p1 is not
    wizAi's seat 1.
    """
    import re

    blanks = dict(unconfigured(source))
    if not blanks:
        return source, []
    filled = []
    for index, (var, school_var) in enumerate(zip(ACCOUNT_VARS, SCHOOL_VARS)):
        if index >= len(wizards):
            break
        name, school = wizards[index]
        for target, value in ((var, name), (school_var, school)):
            placeholder = blanks.get(target)
            if not placeholder or not value:
                continue
            # ...in the script's own spelling. See `script_spelling`.
            value = script_spelling(source, target, value, placeholder)
            # The `var` line only. The guards elsewhere compare against
            # the placeholder literal and MUST keep doing so -- that is
            # how the script tests whether it was configured, and
            # rewriting them would turn every guard permanently false.
            pattern = rf'(^\s*var\s+{re.escape(target)}\s*=\s*)"' \
                      rf'{re.escape(placeholder)}"'
            source, n = re.subn(pattern, lambda m: f'{m.group(1)}"{value}"',
                                source, count=1, flags=re.MULTILINE)
            if n:
                filled.append((target, value))
    return source, filled


def solo_source(source: str):
    """(source, reset) — put the account settings back to placeholders.

    The solo-pilot mode runs a party script over ONE wizard, and the
    scripts were built for exactly that degenerate case. TTS Arc 1's
    own settings comment: "If you're solo questing, you do not have to
    change any of the account information below." Every multi-client
    branch is behind `if NOT Questee2 = "<placeholder>"` guards, so the
    placeholder IS the off switch.

    The complication is that `configure()` may already have filled real
    names in — the source that reaches a run is the dialog's output, not
    the file on disk. The placeholder is recoverable anyway: the guard
    comparisons still hold it. This is `unconfigured`'s assigned-AND-
    compared rule read backwards — find what the script compares the
    variable against, and write that back onto the `var` line.
    """
    import re
    from collections import Counter

    reset = []
    for name in ACCOUNT_VARS + SCHOOL_VARS:
        # Every comparison of this variable against a literal, excluding
        # the assignment itself ('var ' before the name).
        guards = re.findall(rf'(?<!var ){re.escape(name)}\s*=\s*"([^"]*)"',
                            source or "")
        if guards:
            placeholder = Counter(guards).most_common(1)[0][0]
        else:
            # The 2026-08 template has no guard comparisons at all --
            # its multi-client branches are gated on `playercount`
            # instead -- so the placeholder is not recoverable from the
            # text. It does not need to be: the template's own literals
            # are stable, and writing one back is still the right reset
            # (a solo VM's playercount already keeps the p2..p4
            # branches cold; the placeholder keeps any name-keyed path
            # cold too, same as the file ships).
            placeholder = ("SchoolGoesHere" if name in SCHOOL_VARS
                           else "QuestingAccountName")
        new, n = re.subn(rf'(^\s*var\s+{re.escape(name)}\s*=\s*)"[^"]*"',
                         lambda m: f'{m.group(1)}"{placeholder}"',
                         source, count=1, flags=re.MULTILINE)
        if n and new != source:
            source = new
            reset.append(name)
    return source, reset


def preamble_calls(source: str):
    """The one-time `call`s a restart should not repeat, by name.

    A deimoslang program is a pile of `block` definitions, then a few
    top-level statements, then the `loop` it spends the run in. The
    statements before that loop are the program's SETUP: they run once
    when the program starts, and the loop is its steady state. Restart
    the program and they run again -- which is correct for a first
    start and wrong for every restart after it, because the setup has
    already been done.

    All six TTS presets have the same two, and the operator watched
    the second one cost a settings tour every time wizAi reloaded a
    wedged script::

        call First_Run              # prints the banner
        call First_Run_Settings     # opens the game menu, sets 800x600,
                                    # fullscreen off, chat both, ... , OK
        loop { ... }

    Only top-level calls (no indentation) count -- a `call` inside a
    block is part of a definition, not a statement, and the presets
    have thousands of those.

    Nothing here decides anything: `restart_source` applies the rule
    and this only reports it, so a caller can say what it skipped.
    """
    import re

    names = []
    for line in (source or "").splitlines():
        if re.match(r"^(loop|while|until)\b.*\{", line):
            break                        # the steady state starts here
        found = re.match(r"^call\s+([A-Za-z_]\w*)\s*$", line)
        if found:
            names.append(found.group(1))
    return names


def _block_body(source: str, name: str):
    """The lines of `block <name> { ... }`, or None if it is not there."""
    import re

    lines = (source or "").splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^\s*block\s+{re.escape(name)}\s*\{{", line):
            start = i
            break
    if start is None:
        return None
    depth = 0
    body = []
    for line in lines[start:]:
        depth += line.count("{") - line.count("}")
        body.append(line)
        if depth <= 0:
            break
    return body


def restart_source(source: str):
    """(source, skipped) — the program with its one-time setup removed.

    For RESTARTS only. The first start runs the program as written;
    every restart after it starts from instruction 0 again, and the
    setup it re-runs has already been done. In the presets that is a
    banner and a walk through the game's settings menu -- ~20 clicks
    and a menu the operator watched open "randomly" for no benefit,
    once per reload.

    Conservative twice over. A preamble call is only skipped when its
    block is PURE -- no `var` assignment and no nested `call` -- so a
    setup step that seeds state the loop reads is left alone, and so
    is one whose block cannot be found at all. And the calls are
    COMMENTED, not deleted, so the file keeps its line count and an
    error still points where the author's line numbers say.
    """
    import re

    skipped = []
    for name in preamble_calls(source):
        body = _block_body(source, name)
        if body is None:
            continue                     # cannot see it, will not touch it
        inner = "\n".join(body[1:])
        if re.search(r"\bvar\s+\w", inner) or re.search(r"^\s*call\s",
                                                        inner, re.MULTILINE):
            continue                     # it seeds state; let it run
        # `[ \t\r]*`, not `\s*`: `\s` matches newlines, so a greedy
        # `\s*$` eats the blank line after the call and the rewrite
        # silently shortens the program.
        new, n = re.subn(rf"^call[ \t]+{re.escape(name)}[ \t\r]*$",
                         f"# wizAi: one-time setup, already done -- "
                         f"call {name}",
                         source, count=1, flags=re.MULTILINE)
        if n:
            source = new
            skipped.append(name)
    return source, skipped


#: the sigil tail this hardens: the X-spam that joins the sigil,
#: followed by the preset's fixed bet that one countdown is enough.
_SIGIL_TAIL = re.compile(
    r"(times 30 \{[^{}]*sendkey X, \.1[^{}]*\})"
    r"(\r?\n)([\t ]+)sleep 10"
    r"\2\3call Load")


def steady_sigil(source: str):
    """(source, changed) — hold the wizard on a sigil until it fires.

    The presets enter a dungeon with `Enter_Sigil`: spam X onto the
    sigil, `sleep 10`, `call Load`. The 10 is the countdown — and a bet
    that the counter runs once. It loses whenever a second wizard steps
    on late, because the game RESTARTS the counter for the join: in a
    booster party the booster arrives by wizAi's follow, seconds behind
    the scripted quester, so the reset is the normal case, the sleep
    expires mid-countdown, and the main loop's next safe-area teleport
    yanks the quester off the sigil — the booster follows, and the
    operator watches "the counter starts, then both characters teleport
    away and it never goes in".

    The first cut of this fix looped on the TeamUp window staying
    visible, and lost the same way one layer down: a wizard that has
    JOINED the sigil no longer shows the prompt, so the loop read the
    hidden button as "we zoned" and broke out mid-countdown. So the
    wait now watches the one signal a sigil cannot fake: the preset's
    own `waitforzonechange completion` idiom, which holds until the
    dungeon load actually starts — through any number of counter
    resets — and is bounded at 150s by the VM's waitfor patch, whose
    timeout hook says so in the log if the sigil never fires at all.
    `call Load` then rides out the loading screen as before.

    Idempotent: the rewritten tail no longer matches the pattern.
    Line endings and indentation are taken from the matched text, so
    the presets' CRLF survives.
    """
    def hold(m):
        spam, eol, ind = m.group(1), m.group(2), m.group(3)
        return (
            spam
            + f"{eol}{ind}waitforzonechange completion"
            + f"{eol}{ind}call Load")

    out, n = _SIGIL_TAIL.subn(hold, source)
    return (out, True) if n else (source, False)


def set_pacing(source: str, step=None, dialog=None):
    """(source, changed) — override the script's own pacing settings.

    The TTS presets pace themselves with two settings the author put at
    the top of the file::

        var SpeedDelay = 1     # How long each step of the bot will take
        var DialogDelay = 1    # How long before the bot proceeds after
                               # dialogue

    and sprinkle `sleep $SpeedDelay` after nearly every action. They are
    the operator's knob, and editing a 14,000-line file to turn it is
    not a knob — so the GUI passes the values here and the `var` lines
    are rewritten before the VM ever sees them. `None` means "leave the
    script's own value alone", per setting. deimoslang numbers are
    floats (`tokenizer.py:352`), so 0.5 is as valid as 1.
    """
    import re

    changed = []
    for name, value in (("SpeedDelay", step), ("DialogDelay", dialog)):
        if value is None:
            continue
        want = f"{float(value):g}"
        new, n = re.subn(rf'(^\s*var\s+{name}\s*=\s*)[0-9.]+',
                         lambda m: f"{m.group(1)}{want}",
                         source, count=1, flags=re.MULTILINE)
        if n and new != source:
            source = new
            changed.append((name, want))
    return source, changed


def check(source: str, party_size=None):
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
    # A warning, not a refusal: a solo run does not need the account
    # names, and it is the operator's script to run. But an unconfigured
    # party script cannot regroup itself, and that is worth knowing
    # before rather than after -- see `unconfigured`.
    blanks = unfilled(source, party_size)
    if blanks:
        note += ("\n\n⚠ " + f"{len(blanks)} setting(s) are still at the "
                 f"script's placeholder value: "
                 + ", ".join(f"{n} = \"{v}\"" for n, v in blanks[:6])
                 + (" …" if len(blanks) > 6 else "")
                 + ". The script skips its own friend-teleports while "
                   "these are unset, so a wizard that falls behind cannot "
                   "be pulled back — fill them in before a party run.")
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
    #: longest a hotkey press can be made to wait behind the script.
    #:
    #: Was 0.5, and that number was most of "is there a long delay
    #: between actions/teleports for scripts?" -- yes, and it was ours.
    #: Between every 0.5s burst the service tick slept another 0.5s and
    #: did its reads, so the script ran at roughly 40% duty cycle: every
    #: action the author priced at one second cost two and a half. At
    #: 2.5s a burst (with the tick's sleep shortened while the script is
    #: mid-work, see `LiveWorker._script_step`) the duty cycle is ~85%.
    #: A hotkey waiting 2.5s behind a burst is noticeable but fine --
    #: hotkey actions already queue behind teleports twice that long,
    #: and Stop does not wait at all: `should_stop` is checked between
    #: individual instructions inside the burst.
    SLICE = 2.5
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
        #: which instruction went stale and how, in a form that stays
        #: STABLE across the retries ("ip 214: ValueError: ..."), unlike
        #: `last_error`, whose retry count climbs. The reload backoff
        #: compares consecutive reloads on this.
        self.stale_sig = ""
        self._stuck_ip = None
        self._stuck = 0
        #: the source every restart uses -- the program without its
        #: one-time setup -- built lazily on the first restart, and the
        #: names that were left out of it. See `restart_source`.
        self._restart_source = None
        self.skipped_setup = []

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
            self.stale_sig = f"ip {self._ip()}: step-timeout"
            here = self._at_the_ip()
            self.last_error = (
                f"one instruction ran for {self.STEP_LIMIT:.0f}s without "
                f"finishing"
                + (f" — instruction {self._ip()}, `{here}`"
                   if here else " — a 'waitfor…' whose condition never came")
                + f". Cancelling it leaves the VM half-way through an "
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
                self.stale_sig = f"ip {here}: {self.last_error}"
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

    #: how much of the hung instruction's text is worth carrying into
    #: an export line. A `waitfor` with a long window path is the
    #: common case and the head of it is the identifying part.
    HUNG_INSTRUCTION_CHARS = 120

    def _at_the_ip(self):
        """The instruction the VM is sitting on, as text, or "".

        `_ip` has always been computed and then dropped: `stale_sig`
        carried it for the crash-loop backoff and `last_error` -- the
        string that actually reaches the export -- was a constant. Rev
        a4912b2b reloaded 32 times, about 96 minutes of the run, every
        one of them saying "a 'waitfor…' whose condition never came"
        and none of them saying WHICH.

        `VM.step` reads `self.program[self.current_task.ip]`, so the
        instruction is one index away and its `__repr__` is already the
        kind and its data.
        """
        try:
            ip = self._ip()
            program = getattr(self.vm, "program", None)
            if ip is None or not program or not (0 <= ip < len(program)):
                return ""
            return repr(program[ip])[:self.HUNG_INSTRUCTION_CHARS]
        except Exception:
            return ""

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
        # A restart is not a first start. Whatever the program does
        # before its main loop has already been done, and re-doing it
        # costs the run every time -- in the presets, a walk through
        # the game's settings menu on every reload. Computed once and
        # kept, because it is the source every restart from here on
        # will use. See `restart_source`.
        if self._restart_source is None:
            try:
                self._restart_source, self.skipped_setup = \
                    restart_source(self.source)
            except Exception:
                self._restart_source, self.skipped_setup = self.source, []
        try:
            machine = build_vm(self.clients, self._restart_source)
        except Exception as exc:
            # A rewrite that will not compile is worse than the tour.
            self._restart_source, self.skipped_setup = self.source, []
            try:
                machine = build_vm(self.clients, self.source)
            except Exception:
                self.last_error = f"{type(exc).__name__}: {exc}"
                return False
        self.vm = machine
        self.restarts += 1
        self.finished = False
        self.stale = False
        self.stale_sig = ""
        self.failures = 0
        self._stuck_ip, self._stuck = None, 0
        return True

    def stop(self):
        try:
            self.vm.kill()
        except Exception:
            pass
        self.finished = True


def make_runner(clients, source: str, solo: bool = False):
    """Compile `source` against every client, or raise with a reason.

    `clients` is the whole party. deimoslang scripts address wizards as
    `p1`..`p4` and a VM built over one client answers `None` for the
    rest (`vm.py:135-141`), which surfaces as an `AttributeError` deep
    inside an instruction handler rather than as "this script needs more
    wizards than you gave it". A single client is still accepted -- most
    scripts are solo -- but it is passed as a party of one, not as the
    only thing that exists.

    `solo=True` waives the `@clients` requirement below, and only that.
    It is for the solo-pilot mode, which deliberately runs a party
    script over one wizard AFTER `solo_source` has reset the account
    placeholders -- the script's own guards then skip every p2..p4
    branch, so the sections the requirement protects are unreachable.
    And if the author left one unguarded anyway, the stuck-instruction
    reload (`ScriptRunner.STUCK_AT`) is the backstop, exactly as it is
    for a party of two running a script that names p4.
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
    if not solo and need > len(party):
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
