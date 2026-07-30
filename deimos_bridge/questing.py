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
    """Is a dialogue window up and waiting for a click?"""
    button = await window_from_path(client.root_window, ADVANCE_DIALOG_PATH)
    return button is not None and await _visible(button)


async def dialogue_text(client) -> str:
    window = await window_from_path(client.root_window, DIALOG_TEXT_PATH)
    if window is None:
        return ""
    try:
        return await window.maybe_text()
    except Exception:
        return ""


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


# --------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------
async def advance_dialogue(client, max_clicks: int = 40,
                           settle: float = 0.5):
    """(clicks, reason). Click through dialogue until it stops appearing.

    Bounded rather than looping until quiet: a dialogue that re-opens
    forever (a vendor, a mis-click into the wrong NPC) would otherwise
    hang the run with no way to tell from outside.

    The reason exists because zero clicks had two causes and one story.
    A click that *failed* -- the window moved, another program is over
    the game, mouseless input never activated -- returned the same 0 as
    "there was no dialogue", and the caller printed "no dialogue open"
    at a wizard staring at an open dialogue box. Auto-dialogue then spun
    on it forever in silence, movement blocked, nothing on screen.
    """
    clicks, reason = 0, ""
    async with client.mouse_handler:
        while clicks < max_clicks:
            button = await window_from_path(client.root_window,
                                            ADVANCE_DIALOG_PATH)
            if button is None or not await _visible(button):
                break
            try:
                await client.mouse_handler.click_window(button)
            except Exception as exc:
                reason = (f"found the dialogue but the click failed — "
                          f"{type(exc).__name__}: {exc} (is another window "
                          f"over the game?)")
                break
            clicks += 1
            await asyncio.sleep(settle)
    return clicks, reason


async def teleport_to_quest(client):
    """(ok, reason). Jump to the current quest marker."""
    position, reason = await read_quest_position(client)
    if position is None:
        return False, reason
    try:
        await client.teleport(position)
    except Exception as exc:
        return False, f"teleport failed ({type(exc).__name__}: {exc})"
    return True, ""


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
    if (dx * dx + dy * dy) ** 0.5 > radius:
        return False, "not at the quest marker"
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
        _, why = await advance_dialogue(client)
        if why:
            say(why)
        await wait_until_ready(client)
        if await in_battle(client):
            return True

    position, reason = await read_quest_position(client)
    if position is None:
        say(reason)
        return False

    say("teleporting to the quest marker…")
    ok, reason = await teleport_to_quest(client)
    if not ok:
        say(reason)
        return False

    await asyncio.sleep(settle)
    await wait_until_ready(client)          # a door starts a zone change

    if await open_dialogue_if_near(client, on_status=say):
        await asyncio.sleep(settle)
    if await in_dialogue(client):
        await advance_dialogue(client)
        await wait_until_ready(client)
    if await in_battle(client):
        return True

    # Arriving is often not enough: sigils, dungeon doors and quest NPCs
    # need an interact. A failed interact is the difference between "the
    # hop is working" and "the wizard is standing on the sigil doing
    # nothing", so it is said rather than discarded.
    ok, why = await press_x(client)
    if not ok and why:
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
            await advance_dialogue(client)
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
            await advance_dialogue(client)
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
