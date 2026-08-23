"""Getting to the next fight without babysitting it.

Collecting live data is bottlenecked on walking to mobs, not on combat.

**Scope, stated plainly.** This is not Deimos's auto-questing. That is a
large system — navigation graphs, sigils, dungeon logic, zone
traversal — and it lives in `Deimos/src/questing.py`, which pulls in
`src.sprinty_client`, `src.utils`, `src.paths`, `thefuzz`, wizsprinter
and most of the rest of Deimos with it. What is here is built on plain
wizwalker and aims at one thing: keep landing in fights.

What it does, and the traps each part exists to avoid:

  * **Teleport to the quest marker.** `client.quest_position` is a
    first-class wizwalker hook. But `activate_all_hooks` warns that "the
    quest hook is not written if the quest arrow is off"
    (`memory/handler.py:187`), so a wizard with the arrow disabled reads
    a stale or empty position and every teleport silently goes nowhere.
    `read_quest_position` reports that as a *reason* instead of a bare
    failure.

  * **Wait out loading screens.** Teleporting often triggers a zone
    change, and every read taken during the load is meaningless. A hop
    loop that does not wait for the load simply fails on the far side of
    every door.

  * **Retry rather than give up.** The first version returned on the
    first failed teleport, so one transient read during a zone change
    ended the whole hunt and the run then sat waiting for a fight that
    was never going to start.

  * **Press X.** Arriving at the marker is often not enough — sigils,
    dungeon doors and quest NPCs all need an interact.
"""
import asyncio

#: `src/paths.py:30,32,45` — the dialogue advance button, its text area,
#: and the "press X" prompt the game shows when something is interactable.
ADVANCE_DIALOG_PATH = ["WorldView", "wndDialogMain", "btnRight"]
DIALOG_TEXT_PATH = ["WorldView", "wndDialogMain", "txtArea", "txtMessage"]
NPC_RANGE_PATH = ["WorldView", "NPCRangeWin"]
#: `src/paths.py:70` -- the quest-picker an NPC with several things to
#: say puts up INSTEAD of an ordinary dialogue. It has no `btnRight`, so
#: `advance_dialogue` found nothing to click and reported no dialogue,
#: while the window sat there blocking movement. wizwalker's own
#: `Client.is_in_dialog` counts it (`client.py:367-372`) and wizAi did
#: not. Closing it is the unblocking move: whatever wanted it open can
#: open it again, and a run that cannot walk is over.
SERVICES_EXIT_PATH = ["WorldView", "NPCServicesWin", "wndDialogMain", "Exit"]

#: The modal message-box family — the third kind of blocking window,
#: and the one nothing DETECTED. "Do you wish to be transported?" when
#: teleporting to a friend inside a dungeon, "your friend is busy",
#: the zone-load retry prompt: all of them park the wizard exactly like
#: a dialogue, none of them lives under `wndDialogMain`, and a follower
#: staring at one read as "no dialogue open" — which is the operator's
#: "sometimes the bot doesnt realize it's in dialogue and gets stuck
#: waiting too long". `close_menus` knew these buttons; `in_dialogue`
#: and `advance_dialogue` did not. The buttons clicked are the same
#: ones Deimos's own questing clicks in its follower flows
#: (`src/questing.py:227,627,740`).
MODAL_BUTTON_PATHS = (
    ["MessageBoxModalWindow", "messageBoxBG", "messageBoxLayout",
     "AdjustmentWindow", "Layout", "centerButton"],
    ["MessageBoxModalWindow", "messageBoxBG", "messageBoxLayout",
     "AdjustmentWindow", "Layout", "rightButton"],
    ["MessageBoxModalWindow", "messageBoxBG", "messageBoxLayout",
     "AdjustmentWindow", "RetryBtn"],
    ["WorldView", "PlanningPhase", "MessageBoxModalWindow", "messageBoxBG",
     "messageBoxLayout", "AdjustmentWindow", "Layout", "centerButton"],
    ["WorldView", "PlanningPhase", "MessageBoxModalWindow", "messageBoxBG",
     "messageBoxLayout", "AdjustmentWindow", "Layout", "rightButton"],
)
#: `src/paths.py:35` -- the quest tracker's goal line, e.g. "Defeat
#: Krokopatra". The empty string is a real unnamed window in the tree,
#: not a wildcard; Deimos matches it exactly and so does
#: `window_from_path`.
QUEST_GOAL_PATH = ["WorldView", "windowHUD", "QuestHelperHud",
                   "ElementWindow", "", "txtGoalName"]


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------
async def window_from_path(root, path):
    """Walk a named window path, or None.

    wizwalker exposes `children()` and `name()` but no path lookup, so
    this is the same recursive walk Deimos uses (`src/utils.py:56-69`,
    credited there to sirOlaf).
    """
    async def walk(window, remaining):
        if not remaining:
            return window
        try:
            children = await window.children()
        except Exception:
            return None
        for child in children:
            try:
                if await child.name() == remaining[0]:
                    found = await walk(child, remaining[1:])
                    if found is not None:
                        return found
            except Exception:
                continue
        return None

    return await walk(root, list(path))


async def _visible(window):
    try:
        return bool(await window.is_visible())
    except Exception:
        return False


async def _safe(coro_fn, default=False):
    try:
        return await coro_fn()
    except Exception:
        return default


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
async def in_dialogue(client) -> bool:
    """Is a window up that blocks the wizard until it is dismissed?

    All three kinds, because all three block. The ordinary conversation
    box has a `btnRight` to page through; an NPC with several quests on
    offer puts up `NPCServicesWin` instead; and the modal message boxes
    -- the dungeon-teleport confirm above all -- live under
    `MessageBoxModalWindow` with a confirm button and nothing else.
    Each of these was, at some point, the one kind not checked, and a
    wizard parked at the unchecked kind reads as "no dialogue open" --
    which is the state a run dies in.
    """
    button = await window_from_path(client.root_window, ADVANCE_DIALOG_PATH)
    if button is not None and await _visible(button):
        return True
    exit_button = await window_from_path(client.root_window,
                                         SERVICES_EXIT_PATH)
    if exit_button is not None and await _visible(exit_button):
        return True
    return await _visible_modal(client) is not None


async def _visible_modal(client):
    """The first visible modal confirm button, or None."""
    for path in MODAL_BUTTON_PATHS:
        try:
            window = await window_from_path(client.root_window, path)
            if window is not None and await _visible(window):
                return window
        except Exception:
            continue
    return None


async def dialogue_text(client) -> str:
    window = await window_from_path(client.root_window, DIALOG_TEXT_PATH)
    if window is None:
        return ""
    try:
        return await window.maybe_text()
    except Exception:
        return ""


async def read_quest_goal(client) -> str:
    """The quest tracker's current goal line, or "" if it will not read.

    What two wizards questing together have to agree on. The script
    drives both from one program, so when one of them misses a dialogue
    or loses a teleport its GAME quest state falls behind while the
    program marches on -- and from then on every instruction aimed at
    that wizard is aimed at the wrong step. Nothing was reading this, so
    the divergence was invisible until a human noticed the two windows
    doing different things.

    Empty on any failure, deliberately: this feeds a comparison between
    seats, and "could not read" must never masquerade as "on a different
    quest". `goals_agree` treats an empty read as no evidence.
    """
    window = await window_from_path(client.root_window, QUEST_GOAL_PATH)
    if window is None:
        return ""
    try:
        return strip_markup(await window.maybe_text())
    except Exception:
        return ""


def is_collect_goal(goal) -> str:
    """The goal line if this is a Collect step, else "".

    A Collect step is the one quest shape that publishes **no quest
    position at all**, and it does so by design: the game has no single
    place to point the arrow, because the collectibles are scattered
    ground spawns. The operator, after two runs misdiagnosed as a dead
    quest hook: "there is no quest marker, this is a collect quest
    still even when you select on the quest".

    The TTS quester agrees in its own words. Its handling for
    Krokotopia #31 opens `if p1 tracking_quest "get some bling"` with
    the comment `print "Broken quest detected."`, and then does the
    only thing that works -- hardcoded coordinates::

        if p1 inzone Krokotopia/KT_Krokosphinx/KT_ChampHall {
            p1 tp XYZ(-15546.16, 5727.71, 0.0)
            times 10 { p1 sendkey X, .1 }
            ...
            p1 tp XYZ(-20450.05, -9653.53, -9.15e-05)

    So every wizAi rung that aims at a marker is out of the game for
    these steps, and must say so honestly rather than blaming the
    arrow: the arrow is on, and there is nothing for it to point at.

    Matched on the word `collect`, which is the game's own phrasing for
    the whole family and the same test `_maybe_realm_hop` was already
    keyed on -- the two must never disagree about what a Collect step
    is, since one of them decides that the other is allowed to run.
    """
    import re

    text = strip_markup(goal)
    return text if re.search(r"\bcollect\b", text, re.IGNORECASE) else ""


def goal_counter(goal):
    """(what, done, total) for ANY goal carrying "(n of m)", or None.

    The shape, without the Collect test. A counter says the step is a
    tally of several of something -- three minions, four gemstones --
    and that is a fact about the step whatever verb opens it.

    `Defeat Haunted Minion in Triton Avenue (2 of 3)` is the case that
    wanted this: a street kill-count, where the marker sits on a mob
    pack that respawns, and rev d91c3e24's party spent the back half of
    its run holding the script at one because nothing could tell that
    goal from a dungeon boss's. See `LiveWorker._maybe_count_hold`.
    """
    import re

    text = strip_markup(goal)
    m = re.search(r"\(\s*(\d+)\s+of\s+(\d+)\s*\)", text)
    if not m:
        return None
    what = text[:m.start()].strip().lower()
    return what, int(m.group(1)), int(m.group(2))


def collect_count(goal):
    """(what, done, total) for a Collect step, or None.

    "Collect Gemstones in Hall of Champions (0 of 4)" -> the text before
    the counter, 0, 4. The counter is the only progress signal a Collect
    step has -- it publishes no quest position (see `is_collect_goal`),
    so nothing else can tell "picking them up slowly" from "never found
    them at all".
    """
    text = strip_markup(goal)
    if not is_collect_goal(text):
        return None
    return goal_counter(text)


def strip_markup(text) -> str:
    """The goal line as a person would read it.

    The game returns its own markup in the tracker text, so every goal
    in the live logs reads
    `<center>Talk To Mortis in Nightside</center>`. Harmless to the
    comparison -- both sides carry the same tags -- but this string is
    written into the questing log once a minute per wizard and read by a
    human afterwards, and the tags are most of its width.
    """
    import re

    return re.sub(r"<[^>]*>", "", text or "").strip()


def goals_agree(goals) -> bool:
    """Do the wizards that COULD be read agree on the quest goal?

    Unreadable seats are dropped rather than counted as disagreeing, and
    fewer than two readable goals is agreement by default -- a party of
    one is never out of step with itself, and a run where the tracker is
    hidden must not report a desync every tick.
    """
    known = {g for g in goals if g}
    return len(known) < 2


async def wait_until_ready(client, timeout: float = 30.0,
                           poll: float = 0.4) -> bool:
    """Block while the client is in a loading screen.

    Every read taken mid-load is meaningless, so this gates the hop loop.
    Returns False on timeout rather than raising — a stuck load should
    end the hunt cleanly, not the run.
    """
    waited = 0.0
    while waited < timeout:
        if not await _safe(client.is_loading, False):
            return True
        await asyncio.sleep(poll)
        waited += poll
    return False


async def read_quest_position(client):
    """(XYZ or None, reason).

    The reason matters: the commonest cause of "teleport does nothing" is
    the in-game quest arrow being switched off, which leaves the quest
    hook unwritten (`memory/handler.py:187`) and makes every read look
    like an unremarkable failure.
    """
    try:
        position = await client.quest_position.position()
    except Exception as exc:
        return None, (f"could not read the quest position ({type(exc).__name__}) "
                      "— is the in-game quest arrow switched on?")
    if position is None:
        return None, ("no quest position — is the in-game quest arrow "
                      "switched on, and a quest selected?")
    # A quest hook that was never written reads as the origin.
    if all(abs(getattr(position, axis, 0.0)) < 1e-6 for axis in ("x", "y", "z")):
        return None, ("the quest position reads as (0,0,0), which means the "
                      "quest hook was never written — switch the in-game "
                      "quest arrow on and pick a quest")
    return position, ""


async def read_quest_name(client) -> str:
    """The tracked quest's NAME, or "" if it will not read.

    Different from `read_quest_goal`, and needed alongside it. The goal
    is the step ("Talk To Professor Winthrop in Altar of Kings"); the
    name is the quest that step belongs to ("Back to Winthrop").

    Comparing two wizards needs the name. "Talk to Professor Winthrop"
    is the objective of NINE Krokotopia quests spanning main #2 to main
    #19, so a comparison built on goal text can put a wizard seventeen
    steps from where it is -- and `_catch_up` MOVES wizards, so a wrong
    answer drags one backwards through work it has already done. See
    `questlist`.

    Read the way `VM._fetch_tracked_quest_text` reads it, deliberately:
    both sides of a comparison should come out of the same field, and a
    Deimos update that changes how the tracked quest resolves should
    move this with it.

    Empty on any failure, like `read_quest_goal`, and for the same
    reason: this feeds a comparison between seats, where "could not
    read" must never look like "on a different quest".
    """
    try:
        tracked = await client.quest_id()
        manager = await client.quest_manager()
        for quest_id, quest in (await manager.quest_data()).items():
            if quest_id != tracked:
                continue
            key = await quest.name_lang_key()
            # A key that is not a langcode is answered as itself, the
            # same guard the VM's `_lang_name` patch applies. The known
            # literal is "Quest Finder" -- the journal's pseudo-entry --
            # and reading it as "" hid the one state that explains a
            # wizard whose quest position never reads: heartbeats said
            # nothing while the tracker was not on a real quest at all.
            if "_" not in (key or ""):
                return strip_markup(key or "").strip()
            try:
                name = await client.cache_handler.get_langcode_name(key)
            except ValueError:
                return strip_markup(key).strip()
            return strip_markup(name).strip()
    except Exception:
        return ""
    return ""


# --------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------
#: `_dialogue_moved` outcomes. The third is the one that matters: a
#: click that did nothing and a click whose effect could not be READ are
#: different facts, and treating them alike is how auto-dialogue came to
#: report forty cleared windows at a box it never touched.
MOVED, STALLED, UNREADABLE = "moved", "stalled", "unreadable"


async def _dialogue_moved(client, before, settle, poll, button=None):
    """Did the click land? MOVED / STALLED / UNREADABLE.

    The click has taken when the page turns or the box closes, and both
    are readable, so there is no reason to sleep a fixed interval and
    hope. This used to be a flat `sleep(settle)` after every click, and
    on a five-window NPC that is two and a half seconds of a wizard
    standing still.

    STALLED means the box is still up, its text READ, and the text is
    the same as before the click -- so the click provably did not take.
    UNREADABLE means the same wait expired without `dialogue_text`
    giving anything to compare, which is not evidence of either. The
    caller may act on the first and must not act on the second.
    """
    # `button` is the advance button the caller just clicked. Checking
    # whether THAT is still visible is one memory read; `in_dialogue`
    # re-walks the whole window tree from the root, and this runs every
    # `poll` for as long as the box is up. The caller has it already, so
    # asking the tree for it again twenty times a second is pure cost.
    async def still_open():
        if button is not None:
            return await _visible(button)
        return await in_dialogue(client)

    readable = bool(before)
    waited = 0.0
    while True:
        await asyncio.sleep(poll)
        waited += poll
        if not await still_open():
            return MOVED                     # the box closed
        now = await dialogue_text(client)
        if now:
            readable = True
        if now != before:
            return MOVED                     # the page turned
        if waited >= settle:
            break
    # STALLED is a claim that the game was given time to answer and did
    # not. Without a settle window there was no such time, so the only
    # honest verdict is "no evidence" however readable the box is.
    return STALLED if (readable and settle > 0) else UNREADABLE


async def dialogue_opened(client, settle: float = 0.6,
                          poll: float = 0.05) -> bool:
    """Wait for the box the press-X just asked for, rather than sleeping.

    The same trick `_dialogue_moved` uses on the other side of the
    conversation, for the same reason: the game answers in its own time
    and the answer is readable, so a flat `sleep(0.6)` after every
    press-X is 0.6s of a wizard standing still whether the box took
    30ms or never came. Worst case is unchanged -- it falls through to
    the full `settle` -- and the common case is one poll interval.
    """
    waited = 0.0
    while waited < settle:
        await asyncio.sleep(poll)
        waited += poll
        if await in_dialogue(client):
            return True
    return False


async def advance_dialogue(client, max_clicks: int = 40,
                           settle: float = 0.5, poll: float = 0.05):
    """(clicks, reason). Click through dialogue until it stops appearing.

    Bounded rather than looping until quiet: a dialogue that re-opens
    forever (a vendor, a mis-click into the wrong NPC) would otherwise
    hang the run with no way to tell from outside.

    `settle` is now a CEILING on how long each click is given rather
    than a fixed cost per click -- see `_dialogue_moved`.

    The reason exists because zero clicks had two causes and one story.
    A click that *failed* -- the window moved, another program is over
    the game, mouseless input never activated -- returned the same 0 as
    "there was no dialogue", and the caller printed "no dialogue open"
    at a wizard staring at an open dialogue box. Auto-dialogue then spun
    on it forever in silence, movement blocked, nothing on screen.
    """
    clicks, tries, stalls, reason = 0, 0, 0, ""
    async with client.mouse_handler:
        while tries < max_clicks:
            button = await window_from_path(client.root_window,
                                            ADVANCE_DIALOG_PATH)
            if button is None or not await _visible(button):
                # No page to turn. A modal confirm may be holding the
                # wizard instead -- the dungeon-teleport prompt, the
                # zone-load retry -- and clicking it IS the advance.
                # Back around the loop afterwards, because dismissing
                # one can reveal an ordinary dialogue behind it.
                modal = await _visible_modal(client)
                if modal is not None:
                    try:
                        await client.mouse_handler.click_window(modal)
                        clicks += 1
                        tries += 1
                        await asyncio.sleep(0.3)
                        continue
                    except Exception as exc:
                        reason = (f"found a modal prompt but the click "
                                  f"failed — {type(exc).__name__}: {exc}")
                        break
                # There may still be a quest-picker holding the wizard
                # in place -- closing that IS the advance too, and it
                # is one click, not a conversation.
                closed, why = await _close_services(client)
                if closed:
                    clicks += 1
                elif why:
                    reason = why
                break
            before = await dialogue_text(client)
            try:
                await client.mouse_handler.click_window(button)
            except Exception as exc:
                reason = (f"found the dialogue but the click failed — "
                          f"{type(exc).__name__}: {exc} (is another window "
                          f"over the game?)")
                break
            tries += 1
            outcome = await _dialogue_moved(client, before, settle, poll,
                                            button)
            if outcome == MOVED:
                clicks += 1
                stalls = 0
                continue
            if outcome == UNREADABLE:
                # No evidence either way, so this keeps going exactly as
                # it did before there was a distinction to draw.
                clicks += 1
                continue
            # STALLED. The box is still up and its text has not changed,
            # so the click went nowhere -- `click_window` raises nothing
            # when mouseless input is not hooked or another window is
            # over the game, it simply has no effect. This used to be
            # counted as a cleared window and repeated forty times, half
            # a second each: twenty seconds of "clicking through
            # dialogue…" at a box that never moved, then a report of
            # forty windows advanced. Which is what the operator saw.
            stalls += 1
            if stalls >= 2:
                # The mouse has provably failed twice. Last resort: the
                # keyboard, a DIFFERENT hook -- a dead mouseless cursor
                # and a dead keypress do not fail together. A page that
                # the spacebar turns was a dead click, not a wedged box.
                # Only here, not on the first stall, so a single racy
                # click cannot double-advance past a choice.
                before_key = await dialogue_text(client)
                if await _press_spacebar(client):
                    if await _dialogue_moved(client, before_key, settle,
                                             poll, button) != STALLED:
                        clicks += 1
                        stalls = 0
                        continue
                reason = (f"the dialogue is open and the advance button "
                          f"is there, but two clicks and the spacebar all "
                          f"left the same text on screen — the input is "
                          f"not reaching the game, mouse and keyboard "
                          f"both (is another window over it?)")
                break
    return clicks, reason


async def _close_services(client):
    """(closed, reason). Dismiss the multi-quest picker, if one is up."""
    button = await window_from_path(client.root_window, SERVICES_EXIT_PATH)
    if button is None or not await _visible(button):
        return False, ""
    try:
        await client.mouse_handler.click_window(button)
    except Exception as exc:
        return False, (f"a quest menu is open and blocking movement but "
                       f"the click to close it failed — "
                       f"{type(exc).__name__}: {exc}")
    return True, ""


#: how many times a quest teleport is tried before it is called failed.
#: The quester wraps every one in `times 10 { ... }`; ten is its number
#: and there is no reason to be braver than the script people run.
QUEST_TP_TRIES = 4
#: how close counts as arrived, in the game's units.
ARRIVED_WITHIN = 750.0


async def teleport_to_quest(client, tries: int = None, on_status=None):
    """(ok, reason). Get to the quest marker, with the script's protections.

    The operator's point, and it lands squarely on what this used to be:

        the thing with the questing scripts though is that it fixes (or
        adds protections at the very least) to the quest marker
        teleports which often fail

    This was one bare `client.teleport(position)`. Not even `navmap_tp`
    -- so no nav-mesh route, no midpoint or spiral fallback, none of the
    loading-gate work in `teleport_math` -- and no check that it landed.
    Meanwhile the quester wraps every `tp quest` in

        times 10 { tp quest; call Close_Menus; times 20 { sendkey X } }

    and precedes 116 of its teleports with `tp XYZ(0,100000,0)`. Those
    are not stylistic. A quest teleport fails often enough that a loop
    around it is the normal way to write one.

    So: Deimos's best teleport where it is importable -- `collision_tp`
    since the 3.14 port, `navmap_tp` before it; see `_navmap_tp` --
    retried, menus closed between attempts because a window left open
    blocks the next one and reads as a loading screen, and the
    out-of-bounds bump once a plain attempt has not moved the wizard.
    """
    position, reason = await read_quest_position(client)
    if position is None:
        return False, reason
    tries = QUEST_TP_TRIES if tries is None else tries
    hop = _navmap_tp()
    last = ""
    for attempt in range(max(1, tries)):
        if attempt:
            # Between attempts, not before the first: the common case is
            # a teleport that simply works, and a party of four paying
            # for twenty-four window lookups per hop is not free.
            await close_menus(client, on_status)
            if attempt > 1:
                await _bump(client, position)
        try:
            if hop is not None:
                await hop(client, position)
            else:
                await client.teleport(position)
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            continue
        await asyncio.sleep(0.4)
        arrived, why = await _at(client, position)
        if arrived:
            return True, ""
        last = why
    return False, (f"the quest marker would not take after {tries} "
                   f"attempts — {last}" if last else
                   f"the quest marker would not take after {tries} attempts")


def _navmap_tp():
    """Deimos's best teleport, or None if Deimos is not importable.

    Since the 3.14 port that is `collision_tp`: it solves the zone's
    collision geometry for a landing spot that clears every wall -- the
    lands-in-a-wall, rubber-banded-back class of failure this module's
    retry loop exists to absorb -- and falls back BY ITSELF to
    `navmap_tp` when a zone has no collision data or shapely/numpy are
    not installed. `navmap_tp` remains the answer on a Deimos old
    enough not to have it. Either way the callable carries wizAi's own
    patches: the loading gate, the landed/not answer, and
    `on_teleport_result` into the run's log. Optional, because
    `questing` is the module that is meant to work on plain wizwalker.
    """
    try:
        from .deimos_path import ensure_path
        ensure_path()
        try:
            from src.teleport_math import collision_tp
            return collision_tp
        except ImportError:
            from src.teleport_math import navmap_tp
            return navmap_tp
    except Exception:
        return None


async def _at(client, position, within: float = None):
    """(arrived, why). Did the wizard actually end up there?

    The half that was missing entirely. A teleport that silently does
    nothing and a teleport that worked returned the same answer, so the
    caller retried nothing and reported success.
    """
    within = ARRIVED_WITHIN if within is None else within
    try:
        here = await client.body.position()
    except Exception as exc:
        return False, (f"could not read the position afterwards "
                       f"({type(exc).__name__})")
    try:
        gap = sum((getattr(here, a, 0.0) - getattr(position, a, 0.0)) ** 2
                  for a in ("x", "y", "z")) ** 0.5
    except Exception:
        return False, "the positions would not compare"
    if gap <= within:
        return True, ""
    return False, f"still {gap:,.0f} away and the cutoff is {within:,.0f}"


def keycode_x():
    """wizwalker's `Keycode.X`, or None if wizwalker is not importable.

    Split out as a function so the interact path is testable without
    wizwalker: a test overrides this rather than the whole of `press_x`,
    which keeps the calling logic under test instead of stubbed away.
    """
    try:
        from wizwalker import Keycode
        return Keycode.X
    except Exception:
        return None


def keycode_spacebar():
    """wizwalker's `Keycode.SPACEBAR`, or None if unavailable.

    The keyboard advance for dialogue, and a SECOND input path when the
    mouse click at the advance button does not land. A dead mouseless
    cursor hook and a dead keyboard press do not fail together -- they
    are different hooks -- so a box that survives both is genuinely
    wedged, and one the mouse could not turn but the keyboard could is
    exactly the failure the script's own `times 20 { sendkey X }` hits
    (X starts a conversation; it does not advance a MORE page).
    """
    try:
        from wizwalker import Keycode
        return Keycode.SPACEBAR
    except Exception:
        return None


async def _press_spacebar(client, seconds: float = 0.1) -> bool:
    """Send the dialogue-advance key. True only if it went without error."""
    key = keycode_spacebar()
    if key is None:
        return False
    try:
        await client.send_key(key, seconds)
        return True
    except Exception:
        return False


async def press_x(client, seconds: float = 0.1):
    """(ok, reason). Interact — sigils, dungeon doors and quest NPCs need it.

    Every caller used to throw the answer away, so a wizard that was
    teleporting to the right marker and then failing to interact looked
    exactly like one that was working: the status line said "teleporting
    to the quest marker…" every tick, forever, and the dead step was
    reported nowhere.
    """
    key = keycode_x()
    if key is None:
        return False, ("wizwalker did not provide a keycode for X, so "
                       "interacting is disabled — sigils, dungeon doors and "
                       "quest NPCs all need it")
    try:
        await client.send_key(key, seconds)
        return True, ""
    except Exception as exc:
        return False, (f"could not press X to interact "
                       f"({type(exc).__name__}: {exc}) — sigils, dungeon "
                       f"doors and quest NPCs all need it, so nothing will "
                       f"start")


#: Windows that swallow a wizard, and the button that closes each.
#:
#: Lifted from the `Close_Menus` block of the TTS Arc 1 quester, which
#: calls it 63 times -- once after essentially every `tp quest`. That is
#: not decoration. A quest teleport lands a wizard on an NPC, a shop, a
#: sigil or a reagent, any of which can put a window up; a window up
#: blocks movement, and `client.is_loading()` reports some of them as a
#: LOADING SCREEN, so the next teleport is refused as well. The script
#: has this because without it a quester stops.
#:
#: wizAi had one of these paths (`NPCServicesWin`) and no others, which
#: is why `hop_once` could clear a conversation and still be stuck.
CLOSE_MENU_PATHS = (
    ["WorldView", "NPCServicesWin", "wndDialogMain", "Exit"],
    ["WorldView", "ShoppingReagentWindow", "buyWindow", "exit"],
    ["WorldView", "", "Exit"],
    ["WorldView", "TournamentRanking", "exit"],
    ["WorldView", "mainwindow", "exit"],
    ["WorldView", "main", "exit"],
    ["WorldView", "shopGUI", "buyWindow", "exit"],
    ["WorldView", "NPCTrainingGUI", "Dialog", "ExitButton"],
    ["WorldView", "NPCTrainingGUI", "TrainingSelection", "ExitButton"],
    ["WorldView", "NPCTrainingShadowMagicGUI", "Dialog", "ExitButton"],
    ["WorldView", "ShoppingPetSnackWindow", "buyWindow", "exit"],
    ["WorldView", "wndDialogMain", "btnYes"],
    ["WorldView", "wndDialogMain", "btnRight"],
    # The modal family, shared with `in_dialogue`/`advance_dialogue` so
    # the three can never disagree about what counts as blocking again.
    *MODAL_BUTTON_PATHS,
    ["WorldView", "PetLevelUpWindow", "wndPetLevelBkg", "btnPetLevelClose"],
    ["WorldView", "ClassPicture", "exit"],
    ["WorldView", "HelpHousingTips2", "toolbar", "exit"],
    ["WorldView", "SocialSystemsWindow", "CloseSocialSystemsButton"],
    ["WorldView", "windowHUD", "NewFriendsListWindow", "btnClose"],
    ["WorldView", "", "messageBoxBG", "ControlSprite", "cancelButton"],
)


async def close_menus(client, on_status=None) -> int:
    """Click shut every window in `CLOSE_MENU_PATHS`. Returns how many.

    Only what is visible, and every failure is swallowed per window: the
    point is to leave the wizard able to move, and one window whose
    button has moved must not stop the other twenty-three being closed.

    Deliberately NOT limited to the windows that block movement. The
    script closes all of them because the cost of closing one that was
    harmless is nothing, and the cost of missing one is a wizard that
    stands still for the rest of the run.
    """
    closed = 0
    try:
        async with client.mouse_handler:
            for path in CLOSE_MENU_PATHS:
                try:
                    window = await window_from_path(client.root_window, path)
                    if window is None or not await _visible(window):
                        continue
                    await client.mouse_handler.click_window(window)
                    closed += 1
                    await asyncio.sleep(0.15)
                except Exception:
                    continue
    except Exception:
        return closed
    if closed and on_status:
        on_status(f"closed {closed} window(s) that were blocking movement")
    return closed


#: Somewhere no zone contains. The quester uses `tp XYZ(0,100000,0)` 116
#: times, always immediately before a real teleport, and it is not
#: superstition: the game clamps the wizard back to the nearest valid
#: point, which forces a position update the next teleport can act on. A
#: teleport issued from a position the client thinks is already correct
#: is the one that silently does nothing.
NOWHERE = (0.0, 100000.0, 0.0)


async def _bump(client, like) -> bool:
    """Shove the wizard out of bounds so the next teleport takes.

    Built from the type of a position we already hold rather than by
    importing `wizwalker.XYZ`. The first version imported it, and in any
    environment where that import fails the bump did nothing at all --
    silently, and only in the case it exists for.
    """
    try:
        target = type(like)(*NOWHERE)
    except Exception:
        try:
            from wizwalker import XYZ
            target = XYZ(*NOWHERE)
        except Exception:
            return False
    try:
        await client.teleport(target)
        await asyncio.sleep(0.4)
        return True
    except Exception:
        return False


async def press_x_until(client, done, tries: int = 20,
                        gap: float = 0.15):
    """(pressed, reason). Press X until `done()` says so, or `tries` runs out.

    One press was what wizAi did, and the script presses up to twenty --
    `times 20 { sendkey X, .1 }` around a `windowvisible` check, 185
    times across the program. The reason is that X is a proximity
    interact: it does nothing until the wizard has actually finished
    arriving, and after a teleport that can be a second or two of
    settling. A single press lands in the gap and the step never starts.
    """
    key = keycode_x()
    if key is None:
        return 0, ("wizwalker did not provide a keycode for X, so "
                   "interacting is disabled — sigils, dungeon doors and "
                   "quest NPCs all need it")
    pressed = 0
    for _ in range(max(1, tries)):
        try:
            if await done():
                return pressed, ""
        except Exception:
            pass
        try:
            await client.send_key(key, 0.1)
            pressed += 1
        except Exception as exc:
            return pressed, (f"could not press X to interact "
                             f"({type(exc).__name__}: {exc})")
        await asyncio.sleep(gap)
    try:
        if await done():
            return pressed, ""
    except Exception:
        pass
    return pressed, ""


async def in_battle(client) -> bool:
    return await _safe(client.in_battle, False)


async def near_interactable(client) -> bool:
    """Is the game showing its "press X" prompt?

    `NPCRangeWin` is the window that appears when a sigil, dungeon door
    or quest NPC is in range (`src/paths.py:45`).

    Deliberately broad -- see `at_a_sigil` for the narrow question. Two
    callers want breadth: auto-dialogue, which talks to whatever is in
    range, and the walk-through, which is asking "is there anything
    here at all". A caller that is going to FREEZE THE SCRIPT wants the
    other one.
    """
    window = await window_from_path(client.root_window, NPC_RANGE_PATH)
    return window is not None and await _visible(window)


#: `src/paths.py:41`. The Team Up button lives INSIDE `NPCRangeWin`, and
#: only a dungeon sigil puts it there -- a quest NPC, a vendor, the bank
#: and another player's housing object all raise the same range window
#: with no way to team up through it.
SIGIL_PATH = ["WorldView", "NPCRangeWin", "imgBackground", "TeamUpButton"]


async def at_a_sigil(client) -> bool:
    """Is the press-X prompt a DUNGEON SIGIL's, rather than anything's?

    `near_interactable` cannot tell those apart -- `at_quest_marker`'s
    own docstring says so: the range window "appears for *every*
    interactable in range: vendors, bank, the dye shop, other players'
    housing objects". That is fine for clicking dialogue and wrong for
    deciding to freeze the script for 45 seconds.

    Rev 09a0af80 is the cost of not distinguishing them. Five of that
    run's ten quest steps were `Talk To` steps; every one of them put a
    wizard within `AT_THE_MARKER` of an NPC that raised the range
    window, the guard read a partner's prompt as a sigil mid-entry, and
    36% of a 25-minute run went into countdown holds at conversations.

    The discriminator was already in the tree and never used: Deimos
    reads `TeamUpButton` out of the same window (`src/paths.py:41`), and
    the preset quester's own dungeon test is `NPCRangeWin` +
    `TeamUpButton` visible -- the pair, not the first alone.
    """
    window = await window_from_path(client.root_window, SIGIL_PATH)
    return window is not None and await _visible(window)


#: How close to the quest marker counts as "this is the quest NPC".
#: Wizard101's world units put a conversation range at roughly 200-400,
#: and quest markers sit on the objective rather than beside it, so this
#: is generous enough to survive a marker on the far side of an NPC and
#: tight enough to exclude the next vendor along.
#:
#: Seven hundred and fifty was not generous enough. Rev c1d4981c logged
#: `stuck-detail` five times reading "standing 831 away and the cutoff
#: is 750" -- a wizard essentially on its objective, refused by eighty-
#: one units, with auto-dialogue's press and `_maybe_count_hold` both
#: gated behind this. `desperate-hop` took six minutes to close it.
#:
#: This is not the range test that decides whether an interact can
#: happen; the game's own press-X prompt is, and every caller checks
#: `near_interactable` first. This only decides whether the thing in
#: range is the QUEST's -- so the cost of being wide is greeting a
#: vendor standing within a thousand units of the objective, and the
#: cost of being narrow is a wizard that cannot talk to the NPC it is
#: standing on.
QUEST_RADIUS = 1000.0


async def at_quest_marker(client, radius: float = QUEST_RADIUS):
    """(near, reason). Is the wizard standing at its quest objective?

    The discriminator for "should I talk to this NPC". `NPCRangeWin` --
    the game's press-X prompt -- appears for *every* interactable in
    range: vendors, bank, the dye shop, other players' housing objects.
    Clicking on all of them is how auto-dialogue ended up talking to
    everyone walked past, which is worse than not helping, because each
    unwanted conversation has to be clicked back out of.

    Returns (False, reason) rather than a bare False when the quest
    position cannot be read at all, so the caller can say why it is not
    talking to anyone instead of looking broken.
    """
    quest, reason = await read_quest_position(client)
    if quest is None:
        return False, reason
    try:
        here = await client.body.position()
    except Exception as exc:
        return False, f"could not read your position ({type(exc).__name__})"
    if here is None:
        return False, "could not read your position"
    try:
        dx = here.x - quest.x
        dy = here.y - quest.y
        dz = here.z - quest.z
    except AttributeError:
        return False, "position read back in an unexpected shape"
    # Flat distance. Z is height, and a quest NPC one storey up a ramp is
    # still the quest NPC -- including it would refuse the very cases
    # where standing next to someone is unambiguous.
    away = (dx * dx + dy * dy) ** 0.5
    if away > radius:
        # With the distance in it. "not at the quest marker" is the same
        # sentence whether the wizard is standing on the NPC's toes with
        # a marker that reads oddly, or is genuinely across the zone --
        # and the first of those is auto-dialogue looking broken while
        # working exactly as written.
        return False, (f"not at the quest marker — standing {away:,.0f} "
                       f"away and the cutoff is {radius:,.0f}")
    return True, ""


async def open_dialogue_if_near(client, quest_only: bool = True,
                                on_status=None) -> bool:
    """Start the conversation, rather than waiting for one to appear.

    Auto-dialogue that only clicks an *already open* window still needs a
    human to walk up and press X, which is most of the work. If the
    prompt is showing and no dialogue is up yet, press X to open it.

    `quest_only` gates that on standing at the quest marker. On by
    default: the press-X prompt is shown for every interactable in range,
    so without the gate this greets every vendor and signpost on the way
    past.
    """
    if await in_dialogue(client):
        return False
    if not await near_interactable(client):
        return False
    if quest_only:
        near, _ = await at_quest_marker(client)
        if not near:
            return False
    ok, reason = await press_x(client)
    if not ok and reason and on_status:
        on_status(reason)
    return ok


# --------------------------------------------------------------------------
# the hunt
# --------------------------------------------------------------------------
async def hop_once(client, settle: float = 1.2, on_status=None) -> bool:
    """One hop. Returns True if a fight is now on.

    Exists as its own step because the blocking hunt below cannot be run
    from the live worker's service task -- and running it from the fight
    loop was the bug behind "auto-quest does nothing". That loop parks
    inside `wait_for_combat` for as long as it takes a fight to start, so
    a hunt placed before it got exactly one attempt per fight and then
    sat idle forever. Called once per service tick, this keeps trying.
    """
    def say(message):
        if on_status:
            on_status(message)

    await wait_until_ready(client)
    if await in_battle(client):
        return True

    # Dialogue blocks movement, so clear it before trying to move -- and
    # open it first if the game is offering.
    if await open_dialogue_if_near(client, on_status=say):
        await asyncio.sleep(settle)
    if await in_dialogue(client):
        say("clicking through dialogue…")
        n, why = await advance_dialogue(client)
        if why:
            say(why)
        elif not n:
            say("the dialogue closed on its own")
        await wait_until_ready(client)
        if await in_battle(client):
            return True

    position, reason = await read_quest_position(client)
    if position is None:
        say(reason)
        return False

    say("teleporting to the quest marker…")
    ok, reason = await teleport_to_quest(client, on_status=say)
    if not ok:
        say(reason)
        return False

    await asyncio.sleep(settle)
    await wait_until_ready(client)          # a door starts a zone change

    # Before the interact, not after. A quest teleport lands on NPCs,
    # shops and sigils, and anything it puts on screen blocks movement
    # and reads to `is_loading()` as a loading screen -- so the next hop
    # is refused too. The quester calls `Close_Menus` here 63 times.
    await close_menus(client, say)

    if await open_dialogue_if_near(client, on_status=say):
        await asyncio.sleep(settle)
    if await in_dialogue(client):
        await advance_dialogue(client)
        await wait_until_ready(client)
    if await in_battle(client):
        return True

    # Arriving is often not enough: sigils, dungeon doors and quest NPCs
    # need an interact. X is a proximity key and does nothing until the
    # wizard has finished settling, so one press lands in the gap and
    # the step never starts -- the quester presses up to twenty.
    async def something_happened():
        return await in_dialogue(client) or await in_battle(client)

    pressed, why = await press_x_until(client, something_happened)
    if why:
        say(why)
    await asyncio.sleep(settle)
    await wait_until_ready(client)
    if await in_dialogue(client):
        await advance_dialogue(client)
        await wait_until_ready(client)

    return await in_battle(client)


# --------------------------------------------------------------------------
# the quest book: re-arming a dead quest arrow
# --------------------------------------------------------------------------
#: `src/paths.py:185-187` -- the quest page of the big book. Q opens the
#: book straight to it, four quests per page as `wndQuestInfo{0..3}`,
#: and clicking an entry's goal text SELECTS that quest. Selection is
#: the point: the quest-position hook lives in the code that renders
#: the quest helper arrow, so a wizard whose arrow is off never runs
#: it and the hook's export stays (0,0,0) forever -- which starves
#: wizAi's teleports AND the script's own `tp quest` at once. Picking
#: the quest in the book is the operator's manual cure ("switch the
#: in-game quest arrow on and pick a quest"), and it is the same move
#: Deimos's `select_quest_from_questbook` makes (`src/utils.py:795`).
QUEST_BOOK_LIST_PATH = ["WorldView", "DeckConfiguration", "wndQuestList"]
QUEST_BOOK_ALL_SORT_PATH = ["WorldView", "DeckConfiguration", "wndQuestList",
                            "QuestLogAllButton"]
#: within one `wndQuestInfo{n}` entry, the clickable goal line --
#: Deimos's click target, kept identical so a game UI change breaks
#: both in the same visible way.
QUEST_BOOK_ENTRY_GOAL = ["questInfoWindow", "wndQuestInfo", "txtGoal"]
QUEST_BOOK_SLOTS = 4
#: name fragments that mark the quest list's own "next page" control,
#: and the ones that rule a child out. Discovered rather than
#: hard-coded: `Deimos/src/paths.py` names the four slots and the All
#: sort and nothing else, so no path in this codebase has ever pointed
#: at the real control. `_Book.pager` looks for a plausible one, refuses
#: to click anything that could select a quest, and hands the caller
#: `wndQuestList`'s ACTUAL child names when it finds nothing -- which is
#: how the next export names the control instead of this guessing twice.
QUEST_BOOK_PAGER_HINTS = ("right", "next", "forward", "down", "page")
QUEST_BOOK_PAGER_SKIP = ("left", "prev", "back", "up", "close", "sort",
                         "all")
#: how many pages one search will turn. A wizard mid-arc carries a
#: dozen or two accepted quests, so six pages covers the journal twice
#: over -- and it is a hard stop against a control that redraws the
#: same page forever.
QUEST_BOOK_MAX_PAGES = 6
#: how long the receipt poll waits for the hook to be written after the
#: book closes. The arrow's render loop runs every frame once a quest
#: is selected, so a cure that worked shows up in the first second or
#: two; eight covers a slow frame after a zone load.
REARM_PROOF_WAIT = 8.0


def keycode_q():
    """wizwalker's `Keycode.Q`, or None if unavailable.

    Q toggles the quest journal -- the book, open to the quest page.
    The same key closes it, which is how Deimos's own
    `select_quest_from_questbook` gets back out.
    """
    try:
        from wizwalker import Keycode
        return Keycode.Q
    except Exception:
        return None


def _rearm_tokens(text):
    """The words that identify a quest, order-free and counter-free.

    "(0 of 4)" is stripped before tokenizing: the HUD tracker and the
    book render the collect counter at different moments, and a match
    that requires the counts to agree fails exactly when another wizard
    just picked something up.
    """
    import re

    text = re.sub(r"\(\s*\d+\s+of\s+\d+\s*\)", " ", (text or "").lower())
    return set(re.findall(r"[a-z0-9']+", text))


async def _subtree_text(window, depth: int = 6, limit: int = 40) -> str:
    """Every readable text under a window, joined.

    The quest-book entry keeps its quest name and its goal line in
    different leaves depending on quest kind, so the match reads all of
    the entry rather than betting on one path.
    """
    texts = []

    async def walk(win, left):
        if left < 0 or len(texts) >= limit:
            return
        try:
            text = strip_markup(await win.maybe_text())
        except Exception:
            text = ""
        if text:
            texts.append(text)
        try:
            kids = await win.children()
        except Exception:
            return
        for kid in kids:
            await walk(kid, left - 1)

    await walk(window, depth)
    return " ".join(texts)


class _Book:
    """The quest journal, driven the one way, for both of its callers.

    Two rules need this UI and they differ only at the edges:
    `rearm_quest_arrow` re-selects the quest the tracker is ALREADY on
    so the arrow's hook gets written, and `select_quest` puts the
    tracker back onto a quest it has wandered off. Which entry to look
    for and which receipt to demand are the differences; Q, the four
    slots, the All sort and getting the book shut again are identical,
    and a second copy of them would be a second place a game UI change
    has to be found.
    """

    def __init__(self, client):
        self.client = client
        self.key = keycode_q()
        #: how many pages one search actually looked at, and the list's
        #: own child window names when no paging control could be found
        #: among them. Both are for the REPORT: "none of these quests
        #: is in the book" means something different after four entries
        #: than after twenty-four, and a caller that says the first
        #: when it means the second writes a wizard off for half an
        #: hour. See `select_quest`.
        self.pages_read = 1
        self.list_children = []

    async def visible(self) -> bool:
        book = await window_from_path(self.client.root_window,
                                      QUEST_BOOK_LIST_PATH)
        return book is not None and await _visible(book)

    async def open(self):
        """(ok, reason). Q toggles the book open to the quest page."""
        if self.key is None:
            return False, "wizwalker did not provide a keycode for Q"
        for _attempt in range(6):
            if await self.visible():
                return True, ""
            try:
                await self.client.send_key(self.key, 0.1)
            except Exception as exc:
                return False, (f"could not press Q to open the quest book "
                               f"({type(exc).__name__}: {exc})")
            await asyncio.sleep(0.4)
        if await self.visible():
            return True, ""
        return False, "the quest book did not open after six Q presses"

    async def close(self) -> bool:
        """Verified, not assumed. A book left open blocks movement and
        can read as a loading screen, so whoever opened it owns closing
        it -- Q first, then ESC, the same fallback order
        `advance_dialogue` uses."""
        for _attempt in range(6):
            if not await self.visible():
                return True
            try:
                await self.client.send_key(self.key, 0.1)
            except Exception:
                break
            await asyncio.sleep(0.4)
        if await self.visible():
            from .realms import keycode_esc
            esc = keycode_esc()
            if esc is not None:
                try:
                    await self.client.send_key(esc, 0.1)
                except Exception:
                    pass
                await asyncio.sleep(0.4)
        return not await self.visible()

    async def entries(self):
        """[(slot, entry window, its text)] for the visible entries."""
        got = []
        for slot in range(QUEST_BOOK_SLOTS):
            entry = await window_from_path(
                self.client.root_window,
                QUEST_BOOK_LIST_PATH + [f"wndQuestInfo{slot}"])
            if entry is None or not await _visible(entry):
                continue
            got.append((slot, entry, await _subtree_text(entry)))
        return got

    async def show_all(self) -> bool:
        """Click the All sort. The book opens on whatever view the
        player last had, and All is the one listing guaranteed to
        contain every accepted quest."""
        sort = await window_from_path(self.client.root_window,
                                      QUEST_BOOK_ALL_SORT_PATH)
        if sort is None or not await _visible(sort):
            return False
        try:
            await self.client.mouse_handler.click_window(sort)
        except Exception:
            return False
        await asyncio.sleep(0.4)
        return True

    async def children(self):
        """[(name, window)] for the quest list's own children.

        The list itself, not the entries: whatever else `wndQuestList`
        holds -- a sort, a scrollbar, a page arrow -- lives here, and
        nothing in this codebase has a path to any of it.
        """
        book = await window_from_path(self.client.root_window,
                                      QUEST_BOOK_LIST_PATH)
        if book is None:
            return []
        try:
            kids = await book.children()
        except Exception:
            return []
        got = []
        for kid in kids:
            try:
                name = await kid.name()
            except Exception:
                name = ""
            got.append((name or "", kid))
        return got

    async def pager(self):
        """(the control that turns to the next page or None, every child
        name on the list).

        The names come back either way, because the miss is the useful
        half: a run that cannot page says so in the export WITH the
        window names it did see, and the next revision points a path at
        the real control instead of guessing a second time.

        Never an entry window. Clicking `wndQuestInfo{n}` SELECTS the
        quest under it, and picking one at random is precisely the
        failure this book work exists to cure -- so entries, the All
        sort and anything that reads as backwards are refused before a
        hint is even considered.
        """
        kids = await self.children()
        names = [n for n, _w in kids if n]
        for name, win in kids:
            low = name.lower()
            if low.startswith("wndquestinfo"):
                continue
            if any(bad in low for bad in QUEST_BOOK_PAGER_SKIP):
                continue
            if not any(hint in low for hint in QUEST_BOOK_PAGER_HINTS):
                continue
            try:
                if not await _visible(win):
                    continue
            except Exception:
                continue
            return win, names
        return None, names

    async def look_for(self, want):
        """(matching entries, everything seen) for token sets `want`.

        Re-sorts to All and looks again when the first page matched
        nothing, because the book opens on whatever the player left it
        on -- and then TURNS PAGES, because four entries is not a
        journal. Returns every match on the page it stops at rather
        than the first: a caller that must not click the wrong quest
        needs to know there were two. The book is left open at the page
        the match is on, which is where the click has to land.

        Rev f27693c2's run 8 is what the missing half cost. The cure
        finally ran at t=2431 against a page reading "Unicorn Way ...
        Firecat Alley ... Cyclops Lane" -- a wizard freshly through the
        journal's Quest Finder carries a page of newly accepted side
        quests, which is exactly the state this is FOR and exactly the
        state where page one holds none of the answers -- concluded the
        Avalon quest it wanted had never been accepted, and stood down
        for another thirty minutes.
        """
        def hits(seen):
            return [(s, e, t) for (s, e, t) in seen
                    if any(w <= _rearm_tokens(t) for w in want)]

        if self.pages_read > 1:
            # A previous candidate paged to the end. The All sort
            # redraws the listing from the top, which is the only way
            # back this UI is known to offer.
            await self.show_all()
            self.pages_read = 1
        seen = await self.entries()
        found = hits(seen)
        if not found and await self.show_all():
            seen = await self.entries()
            found = hits(seen)
        every = list(seen)
        while not found and self.pages_read < QUEST_BOOK_MAX_PAGES:
            control, names = await self.pager()
            if control is None:
                self.list_children = names
                break
            try:
                await self.client.mouse_handler.click_window(control)
            except Exception:
                break
            await asyncio.sleep(0.4)
            page = await self.entries()
            if not page or ([t for (_s, _e, t) in page]
                            == [t for (_s, _e, t) in seen]):
                # No entries, or the same four again: the control is
                # not a pager, or this is the last page. Either way
                # turning it again learns nothing.
                break
            self.pages_read += 1
            seen = page
            every.extend(page)
            found = hits(page)
        return found, every

    async def select(self, entry, slot=0):
        """(ok, reason). ONE click on the entry's goal line.

        One, not Deimos's five-press burst. The first click SELECTS the
        quest and selection redraws the book, so the four after it land
        on whatever the redraw put under that spot -- a coin-flip at
        the journal's own controls. At rev 7d9b6d6b the burst ran at
        t~250 and by t~257 a client's tracked quest read as the literal
        "Quest Finder" pseudo-entry, which crash-looped the script's
        quest checks for the rest of the run. A click that dies in
        flight is caught by the caller's receipt, and the caller's
        cooldown owns the retry.
        """
        button = await window_from_path(entry, QUEST_BOOK_ENTRY_GOAL)
        if button is None:
            return False, (f"the matching entry (slot {slot}) has no "
                           f"clickable goal line — the book's layout "
                           f"has changed")
        try:
            await self.client.mouse_handler.click_window(button)
        except Exception:
            pass
        await asyncio.sleep(0.3)
        return True, ""


async def _book_openable(client):
    """"" when the book can be opened right now, else why not.

    The three states that make opening it pointless or harmful, checked
    before any cooldown is spent -- a refusal is not an attempt.
    """
    if await in_battle(client):
        return "in a fight — the book cannot open"
    if await in_dialogue(client):
        return ("a dialogue is open — it blocks the book, and clearing "
                "it is the dialogue rung's job")
    if not await wait_until_ready(client, timeout=8.0):
        return "still on a loading screen"
    return ""


async def rearm_quest_arrow(client, goal: str = "", name: str = "",
                            on_status=None):
    """(ok, reason). Re-select the tracked quest so the hook gets written.

    The cure for the one stuck state nothing else on the ladder can
    touch: `read_quest_position` returning (0,0,0) because the quest
    arrow is off. Every rescue that MOVES the wizard needs the marker
    -- the desperate hop, the realm change, the catch-up's `hop_once`,
    and the script's own `tp quest` all read the same dead export -- so
    at rev 98b4c50c Konstantin spent a 44-minute run wandering wherever
    navmap pathing toward (0,0,0) took him while every rung watched.

    The move: open the quest book with Q, find the entry matching the
    tracked quest (by the HUD goal line or the quest name, both read
    fine without the hook), click its goal text -- selection turns the
    helper arrow onto that quest -- close the book, and poll the quest
    position for the receipt. Only a MATCHED entry is ever clicked:
    clicking the wrong one would re-track a side quest, which is worse
    than the disease.
    """
    def say(message):
        if on_status:
            on_status(message)

    want = [t for t in (_rearm_tokens(goal), _rearm_tokens(name)) if t]
    if not want:
        return False, ("neither the goal line nor the quest name is "
                       "readable, so there is nothing to match a quest "
                       "book entry against")

    blocked = await _book_openable(client)
    if blocked:
        return False, blocked

    book = _Book(client)
    opened, why = await book.open()
    if not opened:
        return False, why

    try:
        async with client.mouse_handler:
            hit, seen = await book.look_for(want)
            if not hit:
                listing = " · ".join(t for (_s, _e, t) in seen) or "nothing"
                return False, (f"no visible quest book entry matched the "
                               f"tracked quest — the page showed: "
                               f"{listing[:200]}")
            slot, entry, text = hit[0]
            say(f"quest book open — re-selecting \"{text[:60]}\" so the "
                f"quest arrow comes back on")
            clicked, why = await book.select(entry, slot)
            if not clicked:
                return False, why
    finally:
        closed = await book.close()

    if not closed:
        return False, ("clicked the quest, but the book would not close "
                       "on Q or ESC — leaving it is a wizard that cannot "
                       "move, so this attempt is reported rather than "
                       "trusted")

    waited = 0.0
    while waited < REARM_PROOF_WAIT:
        position, _why = await read_quest_position(client)
        if position is not None:
            return True, ""
        await asyncio.sleep(0.5)
        waited += 0.5
    # What the tracker is on NOW is the diagnostic that decides the
    # next move: a real quest name means the arrow is off in the game's
    # own settings; "Quest Finder" means the journal is on its
    # pseudo-entry and no quest is tracked at all.
    now_on = await read_quest_name(client)
    return False, (f"re-selected the quest in the book, but the quest "
                   f"position still does not read after "
                   f"{REARM_PROOF_WAIT:.0f}s — the arrow may be switched "
                   f"off in the game's own settings"
                   + (f"; the tracker now reads {now_on!r}" if now_on
                      else ""))


#: how long the tracked quest name is polled for after the click. The
#: name is read out of the quest manager rather than a render hook, so
#: it updates as soon as selection lands -- but a book that redrew
#: mid-read blanks it once, which is why this polls at all rather than
#: reading once.
SELECT_PROOF_WAIT = 6.0
#: the fewest words a quest name may carry and still be safe to match
#: on. A one-word name subset-matches half the book -- "Bling" appears
#: in the entry text of every quest that mentions it -- and clicking
#: the wrong entry re-tracks the wrong quest, which is the disease this
#: is the cure for.
SELECT_MIN_TOKENS = 2


async def select_quest(client, names, on_status=None):
    """(ok, reason, in_book). Put the tracker back onto a named quest.

    The cure for a LOST main questline, which is a different failure
    from a dead quest arrow even though both end at the same place.
    The arrow being off leaves the hook unwritten and every teleport
    aimed at (0,0,0); the questline being lost leaves the hook written
    perfectly and every teleport aimed at a side quest. Wizard101's
    helper arrow follows whichever quest is SELECTED, so a wizard that
    accepts a side quest -- or has one auto-selected out of a dialogue
    -- has its arrow, wizAi's teleports and the script's own `tp quest`
    all pointed at the side quest from that moment, and the party's
    main-line progress just stops.

    `names` is an ordered list of candidates, best first, because the
    caller has two answers and neither is certain: the wizard's own
    last main-line quest, and the quest the rest of the party is on.
    The book is already open and its entries already read, so trying
    the second costs nothing.

    `in_book` is the third return because the failures need different
    responses. True: the book was READ and this attempt failed for a
    reason another attempt can fix -- a matched entry that would not
    select, a close that did not take, a page that showed no entries at
    all. False: entries were read across every page the list could be
    turned to and none matched -- evidence the quest was never
    accepted, though still not proof, because the paging control is
    found by name rather than by a path anybody has verified, and a
    journal that could not be paged at all says so in the reason.
    None: the book was never looked in, because no candidate was safe
    to match on, and saying "not in the book" about a quest nothing
    looked for would be a lie that writes the wizard off.

    Only an UNAMBIGUOUS match is ever clicked. Two entries matching one
    name means the tokens do not identify a quest here, and clicking
    either would be a coin-flip on the thing this is meant to fix.

    On success `reason` is the candidate that was selected -- which of
    the two answers turned out to be the right one is worth reporting.
    """
    def say(message):
        if on_status:
            on_status(message)

    if isinstance(names, str):
        names = [names]
    wanted = []
    for name in names:
        tokens = _rearm_tokens(name)
        if len(tokens) >= SELECT_MIN_TOKENS and (name, tokens) not in wanted:
            wanted.append((name, tokens))
    if not wanted:
        # 69 of the 2,110 main-line quests are named in one word
        # ("Pieces", "Fallout", "Warbird"), and one word matched
        # against an entry's whole text -- which carries the goal line
        # as well as the name -- picks whatever happens to contain it.
        # Selecting the wrong quest is the failure being cured, so
        # these are refused and named for the operator instead. `None`
        # rather than False: nothing was looked for, so nothing can be
        # said about what the book holds.
        return False, (f"no candidate quest name carries "
                       f"{SELECT_MIN_TOKENS} words to match on "
                       f"({', '.join(repr(n) for n in names) or 'none given'})"
                       f" — one word matches whatever entry happens to "
                       f"contain it, and picking the wrong quest is the "
                       f"failure this is meant to fix"), None

    blocked = await _book_openable(client)
    if blocked:
        return False, blocked, None

    book = _Book(client)
    opened, why = await book.open()
    if not opened:
        return False, why, None

    picked = ""
    try:
        async with client.mouse_handler:
            # One pass over the book per candidate, in order. `look_for`
            # re-sorts to All on a miss, so the second candidate is
            # looking at the full listing either way.
            for name, tokens in wanted:
                hit, seen = await book.look_for([tokens])
                if not hit:
                    continue
                if len(hit) > 1:
                    where = ", ".join(f"slot {s}" for (s, _e, _t) in hit)
                    return False, (f"{name!r} matched {len(hit)} quest book "
                                   f"entries ({where}) — clicking either "
                                   f"would be a guess at the one thing "
                                   f"this is meant to fix"), True
                slot, entry, text = hit[0]
                say(f"quest book open — selecting {name!r} to put the "
                    f"tracker back on the main questline")
                clicked, why = await book.select(entry, slot)
                if not clicked:
                    return False, why, True
                picked = name
                break
            else:
                tried = ", ".join(repr(n) for n, _t in wanted)
                if not seen:
                    # The book opened and showed NO entries at all. That
                    # is a read that failed, not a journal with nothing
                    # in it — every wizard mid-arc has a page of quests.
                    # The 115-min run at rev f2b8101f made this exact
                    # call: "The book showed: nothing", concluded the
                    # quest was never accepted, and wrote the recovery
                    # off on zero evidence. Unreadable is not absent.
                    # `True` — the retryable arm — because a read that
                    # failed is a UI problem, and the next look may
                    # simply work.
                    return False, (f"the quest book opened but its quest "
                                   f"page showed no entries at all — a "
                                   f"read failure, so whether {tried} is "
                                   f"in the book is unknown"), True
                listing = " · ".join(t for (_s, _e, t) in seen)
                how = (f" No control on the quest list looked like a way "
                       f"to the next page, so only the first "
                       f"{QUEST_BOOK_SLOTS} were ever reachable — this is "
                       f"evidence, not proof, and the list's own children "
                       f"are: {', '.join(book.list_children)}"
                       if book.list_children else
                       f" The pager is matched by window name rather than "
                       f"by a verified path, so this is evidence, not "
                       f"proof")
                return False, (f"none of {tried} is among the "
                               f"{len(seen)} quest book entries read "
                               f"across {book.pages_read} page(s)."
                               + how
                               + f". The entries: {listing[:280]}"), False
    finally:
        closed = await book.close()

    if not closed:
        return False, ("clicked the quest, but the book would not close "
                       "on Q or ESC — leaving it is a wizard that cannot "
                       "move, so this attempt is reported rather than "
                       "trusted"), True

    # The receipt: the tracker reading the quest we clicked. Demanded
    # rather than assumed, for the same reason `rearm_quest_arrow`
    # demands its own -- a click that lands on a redrawing book does
    # nothing, silently, and reporting a cure that did not happen is
    # how a run spends the next ten minutes trusting a dead tracker.
    want = _rearm_tokens(picked)
    waited = 0.0
    now_on = ""
    while waited < SELECT_PROOF_WAIT:
        now_on = await read_quest_name(client)
        have = _rearm_tokens(now_on)
        # Either direction. The list and the HUD abbreviate differently
        # -- the data's "Talk to Dr. Gordon Flemming" against the HUD's
        # "Talk To Gordon Flemming" -- and a one-directional check here
        # would report a selection that worked as one that did not.
        if have and (want <= have or have <= want):
            return True, picked, True
        await asyncio.sleep(0.5)
        waited += 0.5
    return False, (f"clicked {picked!r} in the quest book, but the tracked "
                   f"quest still reads "
                   + (f"{now_on!r}" if now_on else "as nothing")
                   + f" after {SELECT_PROOF_WAIT:.0f}s"), True


async def hop_to_next_fight(client, max_hops: int = 25, settle: float = 1.2,
                            on_status=None, should_stop=None) -> bool:
    """Teleport toward the quest until a fight starts.

    One hop is: wait out any loading, read the marker, teleport, settle,
    wait out the loading that teleport may have caused, clear dialogue,
    press X, and look for combat.

    Unlike the first version this does **not** stop at the first failed
    read. A zone change makes several reads fail in a row, and giving up
    there is exactly why the hunt died after the first teleport. It gives
    up only after `max_hops`, or when the reason is one retrying cannot
    fix (the quest arrow being off).
    """
    def say(message):
        if on_status:
            on_status(message)

    def stopping():
        return bool(should_stop and should_stop())

    last_zone = None
    consecutive_failures = 0

    for hop in range(max_hops):
        if stopping():
            return False

        await wait_until_ready(client)
        if await in_battle(client):
            say("fight started")
            return True

        # Dialogue blocks movement, so clear it before trying to move.
        if await in_dialogue(client):
            say("clicking through dialogue…")
            n, why = await advance_dialogue(client)
            # Said out loud. Both of these call sites threw the reason
            # away, so a dialogue the click could not reach announced
            # "clicking through dialogue…" and then reported nothing at
            # all -- which is what "it says it is clearing it and it
            # does not" looks like from the outside.
            if why:
                say(why)
            elif not n:
                say("the dialogue closed on its own")
            await wait_until_ready(client)
            if await in_battle(client):
                say("fight started")
                return True

        zone = await _safe(client.zone_name, None)
        if zone != last_zone:
            if last_zone is not None:
                say(f"zone changed to {zone} — re-reading the quest marker")
            last_zone = zone

        position, reason = await read_quest_position(client)
        if position is None:
            consecutive_failures += 1
            if "quest arrow" in reason and consecutive_failures >= 3:
                # Retrying cannot fix a hook that is never written.
                say(reason)
                return False
            say(f"{reason} — retrying ({hop + 1}/{max_hops})")
            await asyncio.sleep(settle)
            continue
        consecutive_failures = 0

        say(f"teleporting to the quest marker ({hop + 1}/{max_hops})…")
        ok, reason = await teleport_to_quest(client)
        if not ok:
            say(f"{reason} — retrying")
            await asyncio.sleep(settle)
            continue

        await asyncio.sleep(settle)
        # A teleport onto a door starts a zone change; ride it out.
        await wait_until_ready(client)

        if await in_dialogue(client):
            say("clicking through dialogue…")
            n, why = await advance_dialogue(client)
            if why:
                say(why)
            elif not n:
                say("the dialogue closed on its own")
            await wait_until_ready(client)

        if await in_battle(client):
            say("fight started")
            return True

        # Arriving is often not enough: sigils, dungeon doors and quest
        # NPCs need an interact before anything happens.
        ok, why = await press_x(client)
        if not ok and why:
            say(why)
        await asyncio.sleep(settle)
        await wait_until_ready(client)

        if await in_dialogue(client):
            await advance_dialogue(client)
            await wait_until_ready(client)

        if await in_battle(client):
            say("fight started")
            return True

    say(f"no fight after {max_hops} hops — is the quest one that ends in a "
        "fight? Deimos's own questing handles navigation this does not.")
    return False
