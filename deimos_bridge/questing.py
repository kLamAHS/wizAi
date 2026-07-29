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

#: `src/paths.py:30,32` — the dialogue advance button and its text area.
ADVANCE_DIALOG_PATH = ["WorldView", "wndDialogMain", "btnRight"]
DIALOG_TEXT_PATH = ["WorldView", "wndDialogMain", "txtArea", "txtMessage"]


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
                           settle: float = 0.5) -> int:
    """Click through dialogue until it stops appearing.

    Bounded rather than looping until quiet: a dialogue that re-opens
    forever (a vendor, a mis-click into the wrong NPC) would otherwise
    hang the run with no way to tell from outside.
    """
    clicks = 0
    async with client.mouse_handler:
        while clicks < max_clicks:
            button = await window_from_path(client.root_window,
                                            ADVANCE_DIALOG_PATH)
            if button is None or not await _visible(button):
                break
            try:
                await client.mouse_handler.click_window(button)
            except Exception:
                break
            clicks += 1
            await asyncio.sleep(settle)
    return clicks


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
    """Interact — sigils, dungeon doors and quest NPCs all need it."""
    key = keycode_x()
    if key is None:
        return False
    try:
        await client.send_key(key, seconds)
        return True
    except Exception:
        return False


async def in_battle(client) -> bool:
    return await _safe(client.in_battle, False)


# --------------------------------------------------------------------------
# the hunt
# --------------------------------------------------------------------------
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
        await press_x(client)
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
