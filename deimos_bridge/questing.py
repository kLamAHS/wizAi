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
            name = await client.cache_handler.get_langcode_name(key)
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
                reason = (f"the dialogue is open and the advance button "
                          f"is there, but {stalls} clicks left the same "
                          f"text on screen — the click is not reaching "
                          f"the game (is another window over it, or is "
                          f"mouseless input not hooked?)")
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

    So: `navmap_tp` where Deimos is importable, retried, menus closed
    between attempts because a window left open blocks the next one and
    reads as a loading screen, and the out-of-bounds bump once a plain
    attempt has not moved the wizard.
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
    """Deimos's `navmap_tp`, or None if Deimos is not importable.

    Worth reaching for rather than teleporting raw: it carries the nav
    mesh, the midpoint and spiral fallbacks, and wizAi's own patches to
    the loading gate. Optional, because `questing` is the module that is
    meant to work on plain wizwalker.
    """
    try:
        from .deimos_path import ensure_path
        ensure_path()
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
    """
    window = await window_from_path(client.root_window, NPC_RANGE_PATH)
    return window is not None and await _visible(window)


#: How close to the quest marker counts as "this is the quest NPC".
#: Wizard101's world units put a conversation range at roughly 200-400,
#: and quest markers sit on the objective rather than beside it, so this
#: is generous enough to survive a marker on the far side of an NPC and
#: tight enough to exclude the next vendor along.
QUEST_RADIUS = 750.0


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
