"""Changing realms, for the step other players are contesting.

A Collect step in a busy realm is a queue: the collectibles respawn on
a timer and every wizard standing there takes one. The party can be at
the right marker, out of combat, pressing all the right things, and
still make no progress for ten minutes -- which reads as "stuck" and is
actually "crowded". The fix a person applies is the realm list in the
spellbook: hop to a quiet shard, same zone, same spot, empty spawns.

The scan-and-pick approach here is adapted from a Deimos community
contribution (`get_perfect_realms` / `realm_hop_perfect`, shared by its
author after two days of error-free farming with it) -- the page/slot
addressing and the "Perfect" population filter are theirs. What wizAi
adds is the receipts, because this codebase has already paid for every
fire-and-forget click it vendored (see the stack audit): every page
turn is verified by the page NUMBER changing, the slot's NAME is
checked against the realm being aimed for before Go To Realm is
clicked, and the hop itself is confirmed by the loading screen actually
starting -- a click that produced no load did not change realm, and
saying so is the difference between a retry and a silent lie.
"""
import asyncio
import re

from .questing import _visible, window_from_path

#: `src/paths.py:19,23` -- the spellbook, which the realm list lives in.
SPELLBOOK_OPEN_PATH = ["WorldView", "DeckConfiguration"]
SPELLBOOK_CLOSE_PATH = ["WorldView", "DeckConfiguration", "Close_Button"]

#: realm rows per page of the book's list.
REALM_SLOTS = 7
#: how long a page turn gets to show a new page number.
PAGE_TURN_WAIT = 3.0
#: how long Go To Realm gets to start a loading screen before the hop
#: is called not-taken.
HOP_TAKE_WAIT = 6.0
#: how long the realm change's loading screen gets to finish.
HOP_LOAD_WAIT = 45.0


def keycode_esc():
    """wizwalker's `Keycode.ESC`, or None -- same pattern as `keycode_x`."""
    try:
        from wizwalker import Keycode
        return Keycode.ESC
    except Exception:
        return None


def _clean(text) -> str:
    """The realm list wraps its text in markup (`<center>...</center>`)."""
    return re.sub(r"<[^>]*>", "", text or "").strip()


async def _text_of(window) -> str:
    try:
        return _clean(await window.maybe_text())
    except Exception:
        return ""


async def windows_named(root, name, limit=8):
    """Every window with this name, walking the whole tree.

    The realm buttons have no stable path -- Deimos finds them with
    `get_windows_with_name` -- so this is that, over the same
    `children()`/`name()` surface `window_from_path` uses, which keeps
    it testable against the same fixtures.
    """
    found = []

    async def walk(window):
        if len(found) >= limit:
            return
        try:
            if await window.name() == name:
                found.append(window)
        except Exception:
            return
        try:
            kids = await window.children()
        except Exception:
            return
        for kid in kids:
            if len(found) >= limit:
                return
            await walk(kid)

    await walk(root)
    return found


async def _first_named(root, name):
    got = await windows_named(root, name, limit=1)
    return got[0] if got else None


async def _click_named(client, name) -> bool:
    """Click the first window with this name. False when it could not
    even be attempted -- landing is the CALLER's receipt to check."""
    window = await _first_named(client.root_window, name)
    if window is None:
        return False
    try:
        await client.mouse_handler.click_window(window)
        return True
    except Exception:
        return False


async def _open_spellbook(client):
    """(ok, reason). ESC toggles the book, so look before pressing."""
    key = keycode_esc()
    for attempt in range(6):
        book = await window_from_path(client.root_window, SPELLBOOK_OPEN_PATH)
        if book is not None and await _visible(book):
            return True, ""
        if key is None:
            return False, "wizwalker did not provide a keycode for ESC"
        try:
            await client.send_key(key, 0.1)
        except Exception as exc:
            return False, (f"could not press ESC to open the spellbook "
                           f"({type(exc).__name__}: {exc})")
        await asyncio.sleep(0.3)
    return False, "the spellbook did not open after six ESC presses"


async def _close_spellbook(client):
    """Verified close. A book left open blocks movement and can read as
    a loading screen (the audit's `PageFlip` case), so "probably closed"
    is not enough: click, CHECK, and fall back to ESC -- the keyboard
    being the second input path, exactly as in `advance_dialogue`."""
    for _attempt in range(2):
        book = await window_from_path(client.root_window, SPELLBOOK_OPEN_PATH)
        if book is None or not await _visible(book):
            return
        button = await window_from_path(client.root_window,
                                        SPELLBOOK_CLOSE_PATH)
        if button is not None and await _visible(button):
            try:
                async with client.mouse_handler:
                    await client.mouse_handler.click_window(button)
            except Exception:
                pass
            await asyncio.sleep(0.25)
            continue
        key = keycode_esc()
        if key is None:
            return
        try:
            await client.send_key(key, 0.1)
        except Exception:
            return
        await asyncio.sleep(0.25)
    book = await window_from_path(client.root_window, SPELLBOOK_OPEN_PATH)
    if book is not None and await _visible(book):
        key = keycode_esc()
        if key is not None:
            try:
                await client.send_key(key, 0.1)
            except Exception:
                pass


async def _realms_page(client):
    """(current, last) from the list's own "2 / 4" label, or (None, None)."""
    label = await _first_named(client.root_window, "txtRealmPage")
    if label is None:
        return None, None
    text = await _text_of(label)
    if "/" not in text:
        return None, None
    try:
        current, last = text.split("/", 1)
        return int(current.strip()), int(last.strip())
    except ValueError:
        return None, None


async def _page_turned(client, was, wait=PAGE_TURN_WAIT):
    """The new page number once it differs from `was`, or None.

    The receipt for a page-turn click. The community version slept a
    fixed 0.5s and hoped; a click that silently did not land then reads
    the same page twice and files its realms under the wrong number.
    """
    waited = 0.0
    while waited < wait:
        await asyncio.sleep(0.1)
        waited += 0.1
        current, _last = await _realms_page(client)
        if current is not None and current != was:
            return current
    return None


async def _read_slot(client, slot):
    """(name, population) of one realm row, cleaned, or ("", "")."""
    button = await _first_named(client.root_window, f"btnRealm{slot}")
    if button is None:
        return "", ""
    name_elem = await _first_named(button, f"txtRealm{slot}Name")
    pop_elem = await _first_named(button, f"txtRealm{slot}Population")
    name = await _text_of(name_elem) if name_elem is not None else ""
    pop = await _text_of(pop_elem) if pop_elem is not None else ""
    return name, pop


async def scan_realms(client):
    """(realms, reason). Every realm the book lists, in page order.

    Each entry is {"page", "slot", "name", "population"}. An empty list
    always comes with a reason -- "no realms" and "the list would not
    read" are different problems and used to be the same [].
    """
    ok, why = await _open_spellbook(client)
    if not ok:
        return [], why
    try:
        async with client.mouse_handler:
            if not await _click_named(client, "RealmsButton"):
                return [], ("the spellbook is open but the Realms tab "
                            "would not click")
            await asyncio.sleep(0.4)
            current, last = await _realms_page(client)
            if current is None:
                return [], "the realm list's page number would not read"
            realms = []
            for _turn in range((last or 1) + 1):     # bounded, not while True
                for slot in range(REALM_SLOTS):
                    name, pop = await _read_slot(client, slot)
                    if name:
                        realms.append({"page": current, "slot": slot,
                                       "name": name, "population": pop})
                if current >= (last or current):
                    return realms, ""
                await _click_named(client, "btnRealmRight")
                turned = await _page_turned(client, current)
                if turned is None:
                    return realms, (f"the page number stayed at {current} "
                                    f"after clicking next — returning the "
                                    f"{len(realms)} realm(s) already read")
                current = turned
                # The page NUMBER can repaint before the realm rows do;
                # reading the slots in that gap files page N+1's realms
                # under page N's names. A quarter second covers a frame.
                await asyncio.sleep(0.25)
            return realms, ""
    finally:
        await _close_spellbook(client)


def perfect(realms):
    """The realms whose population reads Perfect, in list order."""
    return [r for r in realms
            if (r.get("population") or "").strip().lower() == "perfect"]


async def hop_to_realm(client, page, slot, expect_name=""):
    """(ok, reason). Change to the realm at page/slot.

    `expect_name` is the safety receipt: the list can shift between the
    scan and the hop, and page 2 slot 3 is only meaningful while it
    still shows the realm that was picked. A mismatch refuses rather
    than hopping blind.
    """
    ok, why = await _open_spellbook(client)
    if not ok:
        return False, why
    try:
        async with client.mouse_handler:
            if not await _click_named(client, "RealmsButton"):
                return False, "the Realms tab would not click"
            await asyncio.sleep(0.4)
            current, _last = await _realms_page(client)
            if current is None:
                return False, "the realm list's page number would not read"
            for _step in range(12):                  # bounded either way
                if current == page:
                    break
                was = current
                await _click_named(client, "btnRealmRight" if page > current
                                   else "btnRealmLeft")
                turned = await _page_turned(client, was)
                if turned is None:
                    return False, (f"stuck on page {was} — the page-turn "
                                   f"click is not landing")
                current = turned
                # Same repaint gap as the scan: the slot name checked
                # below must be the new page's, not the old frame's.
                await asyncio.sleep(0.25)
            if current != page:
                return False, f"could not reach page {page} (at {current})"
            if expect_name:
                shown, _pop = await _read_slot(client, slot)
                if shown and shown != expect_name:
                    return False, (f"page {page} slot {slot} now shows "
                                   f"{shown!r}, not {expect_name!r} — the "
                                   f"list moved, so hopping would go "
                                   f"somewhere else. Rescan")
            if not await _click_named(client, f"btnRealm{slot}"):
                return False, f"realm slot {slot} would not click"
            await asyncio.sleep(0.3)
            if not await _click_named(client, "btnGoToRealm"):
                return False, "the Go To Realm button would not click"
            # THE receipt: a realm change is a zone reload. No loading
            # screen means no hop -- the click died, or this is already
            # that realm -- and only this check can tell a landed hop
            # from a silent no-op.
            if not await _loading_started(client):
                return False, ("clicked Go To Realm and no loading screen "
                               "came — the hop did not take (already in "
                               "that realm, or the click died)")
    finally:
        await _close_spellbook(client)
    await _wait_loaded(client)
    return True, ""


async def _loading_started(client, within=HOP_TAKE_WAIT) -> bool:
    waited = 0.0
    while waited < within:
        try:
            if await client.is_loading():
                return True
        except Exception:
            pass
        await asyncio.sleep(0.1)
        waited += 0.1
    return False


async def _wait_loaded(client, limit=HOP_LOAD_WAIT):
    """Bounded, unlike the community version's `while is_loading` --
    a PageFlip window stuck open reads as loading forever."""
    waited = 0.0
    while waited < limit:
        try:
            if not await client.is_loading():
                return
        except Exception:
            return
        await asyncio.sleep(0.2)
        waited += 0.2
