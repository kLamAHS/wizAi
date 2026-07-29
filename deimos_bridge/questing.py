"""Getting to the next fight without babysitting it.

Collecting live data is bottlenecked on walking to mobs, not on combat.
Two small pieces remove most of that: teleport to the quest marker, and
click through the dialogue that follows.

**Scope, stated plainly.** This is not Deimos's auto-questing. That is a
large system — navigation graphs, sigils, dungeon logic, zone
traversal — and it lives in `Deimos/src/questing.py`, which pulls in
`src.sprinty_client`, `src.utils`, `src.paths`, `thefuzz`, wizsprinter
and most of the rest of Deimos with it. Wiring that in would drag the
whole dependency set back, including wizsprinter's Python 3.13 floor,
for a feature Deimos already does better in its own window.

What is here instead is the pair that combat data collection actually
needs, built on plain wizwalker so it costs nothing:

  * `teleport_to_quest` — `client.quest_position` is a first-class
    wizwalker hook (`examples/quest_teleporter.py` does exactly this).
  * `advance_dialogue` — clicking `WorldView/wndDialogMain/btnRight`,
    the same window path Deimos uses (`src/paths.py:30`).

If you want real questing, run Deimos for that and this for the fights.
"""

#: `src/paths.py:30,32` — the dialogue advance button and its text area.
ADVANCE_DIALOG_PATH = ["WorldView", "wndDialogMain", "btnRight"]
DIALOG_TEXT_PATH = ["WorldView", "wndDialogMain", "txtArea", "txtMessage"]


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


async def advance_dialogue(client, max_clicks: int = 30,
                           settle: float = 0.6) -> int:
    """Click through dialogue until it stops appearing.

    Bounded rather than looping until quiet: a dialogue that re-opens
    forever (a vendor, a mis-click into the wrong NPC) would otherwise
    hang the run with no way to tell from outside. Returns how many
    clicks it made.
    """
    import asyncio

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


async def teleport_to_quest(client) -> bool:
    """Jump to the current quest marker.

    Needs the quest and teleport hooks; `activate_hooks()` turns them on,
    so a live run has them already. Returns False rather than raising if
    the position cannot be read — no quest selected, a loading screen,
    a zone with no marker.
    """
    try:
        position = await client.quest_position.position()
    except Exception:
        return False
    if position is None:
        return False
    try:
        await client.teleport(position)
        return True
    except Exception:
        return False


async def is_busy(client) -> bool:
    """In combat, loading, or mid-dialogue — anything that means "do not
    teleport right now"."""
    for check in ("in_battle", "is_loading"):
        fn = getattr(client, check, None)
        if fn is None:
            continue
        try:
            if await fn():
                return True
        except Exception:
            continue
    return await in_dialogue(client)


async def hop_to_next_fight(client, max_hops: int = 8, settle: float = 1.2,
                            on_status=None):
    """Teleport toward the quest and clear dialogue until combat starts.

    One hop is: teleport to the marker, click through whatever dialogue
    that triggers, and check whether a fight began. Bounded by
    `max_hops`, because a quest whose marker is not a fight would
    otherwise spin forever.

    Returns True if combat started.
    """
    import asyncio

    def say(message):
        if on_status:
            on_status(message)

    for hop in range(max_hops):
        try:
            if await client.in_battle():
                return True
        except Exception:
            pass

        say(f"teleporting to the quest marker ({hop + 1}/{max_hops})…")
        if not await teleport_to_quest(client):
            say("no quest position to teleport to")
            return False

        await asyncio.sleep(settle)

        if await in_dialogue(client):
            say("clicking through dialogue…")
            await advance_dialogue(client)
            await asyncio.sleep(settle)

        try:
            if await client.in_battle():
                say("fight started")
                return True
        except Exception:
            pass

    say("gave up hopping — no fight found")
    return False
