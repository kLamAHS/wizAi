"""Keeping four wizards in the same battle circle.

`hivemind.py` makes a party plan one round together. It can only do that
for wizards who are *in the same duel*, and nothing was getting them
there: four clients each running the questing independently walk to four
different places, take four different quests, and coordinate perfectly
with nobody.

So one wizard leads and the rest follow. The leader quests; the followers
do exactly two things:

  **Close the gap.** Out of combat and further than `FOLLOW_RADIUS` from
  the leader, teleport onto them. Inside the same zone that is a plain
  XYZ `client.teleport` -- the same call the wisp sweep and the quest
  hop already use, with no UI to click and nothing to go wrong. Across
  zones it has to be the friends list, because an XYZ teleport cannot
  change zone; that path needs the leader's wizard name and says so when
  it does not have one.

  **Walk into the fight.** When the leader is in a duel and a follower is
  not, arriving beside them is not enough -- Wizard101 puts you in the
  circle only if you touch the sigil or a mob. `SprintyClient.
  tp_to_closest_mob` is exactly that step and Deimos already ships it.

Deliberately *not* a questing implementation. Deimos's own multi-client
questing (`Deimos/src/questing.py`) is a far larger thing -- leader
election, dungeon desync correction, solo-zone detection, friend-code
exchange -- and this does not try to be it. It is the smallest amount of
movement that turns four independent wizards into four wizards in one
fight, which is the input the hivemind needs and the only thing it is
missing.

Everything here is best effort and returns `(did_something, reason)`
rather than raising: a follower that cannot reach its leader must cost
one status line, not the run.
"""


#: How far a follower may drift before it teleports back. Not zero: the
#: game moves wizards apart on its own -- entering a fight, a loading
#: screen, the circle's own seating -- and a follower that re-teleports
#: every tick would spend the fight teleporting instead of fighting.
FOLLOW_RADIUS = 900.0

#: How close a follower must stand once the leader is at a press-X
#: prompt. A dungeon sigil admits exactly the wizards STANDING ON it
#: when its countdown fires -- rev e786b716: the leader entered Knight's
#: Court T2 alone while the booster stood "together" a few hundred units
#: up the street, and only reached the fight a round late through the
#: friends-list teleport. Standing on the sigil restarts the countdown
#: once and puts both wizards in the dungeon -- and the duel -- from
#: round one, which is the booster's whole job.
SIGIL_RADIUS = 200.0

#: How long one friends-list teleport attempt may take.
#:
#: wizwalker's `teleport_to_friend_from_list` opens with
#: `_cycle_to_online_friends`, which is
#:
#:     while (await text()) != "Online Friends":
#:         click(right_button); wait(..., timeout=5)
#:
#: -- an unbounded loop. A wizard whose friends list never reads exactly
#: that string (a different tab layout, a click that does not land, a
#: label the read returns empty) sits in it clicking the page button for
#: the rest of the run. That is not a slow follow, it is a permanent
#: one: the follow step holds this wizard's drive lock while it runs, so
#: every queued teleport, wisp sweep and potion waits behind it and
#: every further keypress is refused as already queued. One unbounded
#: loop in wizwalker took all four hotkeys away.
#:
#: 45s is comfortably more than a working teleport needs -- open the
#: window, page to online friends, click the name, confirm, sit through
#: the animation -- and finite, which is the whole point.
TELEPORT_TIMEOUT = 45.0


async def _safe(coro_fn, default=None):
    try:
        return await coro_fn()
    except Exception:
        return default


async def zone(client):
    return await _safe(client.zone_name, None)


async def position(client):
    body = getattr(client, "body", None)
    if body is None:
        return None
    return await _safe(body.position, None)


def _distance(a, b):
    """Straight-line distance, or None when either end will not read."""
    if a is None or b is None:
        return None
    try:
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2
                + (a.z - b.z) ** 2) ** 0.5
    except AttributeError:
        return None


async def in_battle(client) -> bool:
    return bool(await _safe(client.in_battle, False))


async def at_a_prompt(client) -> bool:
    """Is this client showing the game's press-X prompt?

    Read for the LEADER, from a follower's follow tick: a leader whose
    client shows `NPCRangeWin` is standing at an interactable -- a
    dungeon sigil, most importantly -- and "near the leader" stops
    being good enough for the follower (see `follow`). Best-effort: a
    window that will not read is not a prompt.
    """
    try:
        from . import questing
        return bool(await questing.near_interactable(client))
    except Exception:
        return False


async def wizard_name(client):
    """The wizard's own name, if the client will say.

    Only needed for the cross-zone teleport, which goes through the
    friends list and matches on the name. Deimos reads this off the
    character-select screen, which is not available mid-run -- so this
    tries what the running client exposes and returns None rather than
    inventing one. `LiveWorker` fills the gap from the other direction:
    a combat read already names the client's own member, so the first
    duel supplies it.
    """
    for attr in ("wizard_name", "character_name"):
        value = getattr(client, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if callable(value):
            got = await _safe(value, None)
            if isinstance(got, str) and got.strip():
                return got.strip()
    return None


def helper_followed_name(goal) -> str:
    """The full wizard name in the game's own Quest Helper line, or "".

    Rev 676d6e77's booster stood one zone from its quester for FOUR
    HOURS because the friends-list teleport had no name to aim
    ("wizard name is not known yet ... read from the first duel") --
    while its own goal line read, verbatim, `Quest Helper Following
    Konstantin VeränderungBeschwörer`. The game itself publishes the
    followed wizard's exact full name, umlauts and all, before any
    duel has happened, and it is the one spelling guaranteed to match
    the friends list. Harvesting it is free.
    """
    import re

    if not goal:
        return ""
    m = re.search(r"Quest Helper Following\s+([^\n\r]+)", goal)
    return m.group(1).strip() if m else ""


#: The hardened friends-list teleport, loaded once. See
#: `_hardened_friend_tp`.
_HARDENED = {"fn": None, "tried": False}


def _hardened_friend_tp():
    """wizAi's hardened `teleport_to_friend_from_list`, or None.

    The vendored wizwalker carries the widened window-waits, the
    row-click retry and the bounded Online-Friends cycle -- but the
    RUNNING wizwalker is the pip-installed fork, and
    `wizwalker.extensions.scripting` resolves to the fork's own
    unhardened copy: the overlay in `deimos_path` can only ADD missing
    extension subpackages, never shadow ones the fork ships. So the
    hardened module is loaded from its file, by path, into a throwaway
    namespace. Its imports are all absolute (`wizwalker.memory...`,
    `wizwalker.utils`, `regex`) and resolve against the installed
    wizwalker at exec time, which is exactly right: hardened logic
    over the running primitives. Best-effort -- any failure falls back
    to the stock import, same as before.
    """
    if _HARDENED["tried"]:
        return _HARDENED["fn"]
    _HARDENED["tried"] = True
    try:
        import os

        from .deimos_path import DEIMOS_ROOT

        path = os.path.join(DEIMOS_ROOT, "libs", "wizwalker", "wizwalker",
                            "extensions", "scripting", "utils.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        namespace = {"__name__": "_wizai_hardened_friend_tp"}
        exec(compile(source, path, "exec"), namespace)
        fn = namespace.get("teleport_to_friend_from_list")
        _HARDENED["fn"] = fn if callable(fn) else None
    except Exception:
        _HARDENED["fn"] = None
    return _HARDENED["fn"]


#: resolved full friends-list names, keyed by the short name a duel
#: gives. One lookup per leader per run rather than one per follow.
_FULL_NAMES = {}


async def _read_friends_list(client) -> str:
    """The friends list's raw text, or "". Leaves the window as it found it.

    Opening it and walking away is not free: the friends window blocks
    movement, and a wizard standing behind one cannot quest, follow or
    be teleported. This used to leave it open on every call.
    """
    try:
        from .deimos_path import ensure_path

        ensure_path()
        from wizwalker.extensions.scripting.utils import _maybe_get_named_window
    except Exception:
        return ""

    opened = False
    try:
        root = client.root_window
        try:
            window = await _maybe_get_named_window(root, "NewFriendsListWindow")
        except ValueError:
            button = await _maybe_get_named_window(root, "btnFriends")
            await client.mouse_handler.click_window(button)
            opened = True
            window = await _maybe_get_named_window(root, "NewFriendsListWindow")
        listing = await _maybe_get_named_window(window, "listFriends")
        return await listing.maybe_text() or ""
    except Exception:
        return ""
    finally:
        if opened:
            await _put_the_friends_list_away(client)


#: One friends-list row, as wizwalker's `_friend_list_entry` parses it.
#:
#: A local copy rather than an import, because the import is the part
#: that cannot happen off Windows -- `wizwalker.extensions.scripting.
#: utils` pulls in the memory layer -- and reading a name out of text is
#: not Windows-specific at all. Keeping it here is what lets the whole
#: name resolution be tested. `tests/test_deimos_patches.py` compares
#: the two patterns character for character, so a Deimos bump that
#: changes the game's markup fails there instead of silently matching
#: nothing here.
_ENTRY = (r"<Y;\d+><X;\d+><indent;0><Color;[\w\d]+><left>"
          r"<icon;FriendsList/Friend_Icon_List_0(?P<icon_list>[12])\."
          r"dds;\d+;\d+;(?P<icon_index>\d+)></left><Y;(?P<name_y>[-\d]+)>"
          r"<X;(?P<name_x>[-\d]+)>"
          r"<indent;\d+><Color;[\d\w]+>(<left>)?<COLOR;[\w\d]+>(?P<name>[\w ]+)")


def names_in(text):
    """Every wizard name the friends list text holds, in order."""
    import re

    return [m.group("name") for m in re.finditer(_ENTRY, text or "")]


async def friends_list_names(client, shorts):
    """{short name: full name} for every one of `shorts` on this list.

    One read answers for the whole party, which is the point: a
    three-wizard party needs three full names and each client's list
    holds the other two, so two reads cover everybody rather than six.
    """
    wanted = [s for s in shorts if s]
    found = {s: _FULL_NAMES[s] for s in wanted if s in _FULL_NAMES}
    missing = [s for s in wanted if s not in found]
    if not missing:
        return found

    for name in names_in(await _read_friends_list(client)):
        for short in missing:
            # Prefix, not fuzzy: a duel reports the first name and the
            # list leads with it, and anything looser would happily
            # name the wrong friend.
            if name == short or name.startswith(short + " "):
                _FULL_NAMES[short] = name
                found[short] = name
    return found


async def friends_list_name(follower, short_name):
    """The leader's FULL name as the friends list spells it, or "".

    A combat read gives a wizard's first name -- "Jeffrey" -- and the
    friends list holds "Jeffrey IslandBringer". wizwalker matches on
    `friend_name == name`, exactly, so the teleport could never find a
    leader it was looking at: "Could not find friend with icon None icon
    list None and/or name Jeffrey", forever, while the friends window
    sat open next to it showing exactly one online friend.

    The same exactness is why the quester script has to be given full
    names too: `friendtp Main_Account` ends in the same wizwalker call
    against the same list. See `LiveWorker._resolve_party_names`.

    So the list is read and the entry whose name starts with what a duel
    told us is taken. Prefix rather than fuzzy: first names are what the
    duel reports and what the friends list leads with, and anything
    looser would happily teleport to the wrong friend.
    """
    if short_name in _FULL_NAMES:
        return _FULL_NAMES[short_name]
    return (await friends_list_names(follower, [short_name])).get(
        short_name, "")


async def _put_the_friends_list_away(client):
    """Best effort: close the friends list and the character panel.

    Both block movement, so a wizard left behind them is a wizard that
    cannot follow, cannot quest and cannot be walked out by anything --
    and the next teleport attempt starts by clicking through them.
    That is what turned one failed rejoin into ten in rev cfeb9a85.

    wizwalker's own `teleport_to_friend_from_list` does this in a
    `finally` now; this covers the timeout, which cancels that finally
    before it can run. Never raises.
    """
    try:
        from .deimos_path import ensure_path

        ensure_path()
        from wizwalker.extensions.scripting.utils import _close_friend_windows
    except Exception:
        return False
    try:
        await _close_friend_windows(client)
    except Exception:
        return False
    return True


async def teleport_to_leader_across_zones(follower, leader_name):
    """(ok, reason). The friends-list teleport, for a different zone.

    An XYZ teleport cannot change zone, so this is the only way back to
    a leader who has walked through a door. It drives the real friends
    list with the mouse, which is why it is under `mouse_handler` and
    why it is the slow path used only when the zones actually differ.
    """
    if not leader_name:
        return False, ("the leader is in another zone and its wizard name "
                       "is not known yet, so the friends-list teleport "
                       "cannot pick it out — the name is read from the "
                       "first duel, or walk them into the same zone once")
    try:
        from .deimos_path import ensure_path

        ensure_path()
        # The hardened copy first -- widened waits, row-click retry,
        # bounded Online-Friends cycle -- and the fork's stock one only
        # when it will not load. Which copy ran is stamped into the
        # failure reasons, because rev 676d6e77 spent four hours
        # failing this teleport and exported one line about it.
        teleport_to_friend_from_list = _hardened_friend_tp()
        copy = "hardened copy"
        if teleport_to_friend_from_list is None:
            from wizwalker.extensions.scripting import \
                teleport_to_friend_from_list
            copy = "stock wizwalker"
    except Exception as exc:
        return False, (f"cross-zone follow needs wizwalker's scripting "
                       f"extension ({type(exc).__name__}: {exc})")

    import asyncio

    async def attempt(name):
        """(landed, fatal reason). A name that is simply not on the list
        is neither -- it just means try another spelling.

        Bounded, because the thing it calls is not: see
        `TELEPORT_TIMEOUT`. A timeout is *fatal* rather than "try the
        next spelling" -- whatever went wrong happened before the name
        was ever looked at, so a second spelling would hang the same way
        and the follower would spend three timeouts finding that out.
        """
        try:
            await asyncio.wait_for(
                teleport_to_friend_from_list(follower, name=name),
                TELEPORT_TIMEOUT)
        except asyncio.TimeoutError:
            # The one failure wizwalker's own cleanup cannot cover.
            # `teleport_to_friend_from_list` closes the friends list and
            # the character panel in a `finally` now, but a timeout
            # CANCELS it, and a cancelled coroutine raises again at the
            # first `await` in that finally -- CancelledError is not an
            # Exception, so the suppress inside does not hold it. The
            # windows are therefore left up, on the one path where the
            # wizard is already known to be wedged. Closed from out here
            # instead, where nothing is cancelled.
            await _put_the_friends_list_away(follower)
            return False, (
                f"the friends-list teleport to {name} ran for "
                f"{TELEPORT_TIMEOUT:.0f}s without finishing and was cut "
                f"off ({copy}). It gets stuck paging the list to 'Online "
                f"Friends' when that tab will not come up — check the "
                f"friends window is not already open on another tab, or "
                f"turn 'follow the leader' off and walk this wizard over")
        except ValueError as exc:
            if "Could not find friend" in str(exc):
                return False, ""
            return False, (f"could not teleport to {name} through the "
                           f"friends list (ValueError: {exc}; {copy})")
        except Exception as exc:
            return False, (f"could not teleport to {name} through the "
                           f"friends list ({type(exc).__name__}: {exc}; "
                           f"{copy})")
        return True, ""

    async with follower.mouse_handler:
        tried = []
        # The resolved full name first when we have one, because the
        # exact match is the only one wizwalker does.
        for name in (_FULL_NAMES.get(leader_name), leader_name):
            if not name or name in tried:
                continue
            tried.append(name)
            landed, fatal = await attempt(name)
            if landed:
                _FULL_NAMES[leader_name] = name
                return True, ""
            if fatal:
                return False, fatal

        # Not on the list under any spelling we knew. Read the list and
        # find out what it actually calls them.
        full = await friends_list_name(follower, leader_name)
        if full and full not in tried:
            landed, fatal = await attempt(full)
            if landed:
                _FULL_NAMES[leader_name] = full
                return True, ""
            if fatal:
                return False, fatal

    return False, (f"could not find {leader_name} on this wizard's friends "
                   f"list — they have to be friends and online. A duel "
                   f"reports a first name and the list holds the full one, "
                   f"so the list was searched for a '{leader_name} ...' "
                   f"entry too and there was none")


async def join_the_fight(follower):
    """(ok, reason). Step into the duel the leader is already in.

    Arriving beside the leader is not joining: the game puts you in the
    circle only when you touch a sigil or a mob. This is that touch.
    """
    try:
        from .upkeep import _sprinty

        sprinty = _sprinty(follower)
    except Exception as exc:
        return False, (f"joining the leader's fight needs Deimos's "
                       f"SprintyClient ({type(exc).__name__}: {exc})")
    try:
        if await sprinty.tp_to_closest_mob():
            return True, ""
    except Exception as exc:
        return False, (f"could not step into the fight "
                       f"({type(exc).__name__}: {exc})")
    return False, "no mob in range to step into"


async def follow(follower, leader, leader_name=None,
                 radius: float = FOLLOW_RADIUS):
    """Put one wizard where its leader is. Returns (moved, reason).

    `moved` is False for "nothing to do" as well as for "could not",
    which is why the reason comes back with it: a follower that is
    already there and a follower that cannot read its leader's position
    both look like a follower standing still, and only one of those is
    a problem.

    Ordered by what would go wrong. A follower already in a duel is left
    alone -- it may be in the leader's duel, and teleporting out of a
    fight is not a thing the game lets you do anyway. Then zone, because
    an XYZ teleport across a zone boundary silently does nothing. Then
    distance. Then, only once it has arrived, the sigil step.
    """
    if await in_battle(follower):
        return False, ""

    here, there = await zone(follower), await zone(leader)
    if here is not None and there is not None and here != there:
        ok, why = await teleport_to_leader_across_zones(follower, leader_name)
        if not ok:
            return False, why
        return True, f"followed the leader into {there}"

    target = await position(leader)
    if target is None:
        return False, ("could not read the leader's position, so the party "
                       "cannot regroup")

    gap = _distance(await position(follower), target)
    leader_fighting = await in_battle(leader)
    if gap is not None and gap <= radius and not leader_fighting:
        # "Together" is street-sized -- until the leader stands at a
        # press-X prompt. A sigil only admits the wizards ON it, so a
        # follower keeping a polite 900-unit distance queues for
        # nothing and enters the dungeon a round late (or not at all).
        if gap > SIGIL_RADIUS and await at_a_prompt(leader):
            try:
                await follower.teleport(target)
            except Exception as exc:
                return False, (f"could not join the leader at its prompt "
                               f"({type(exc).__name__}: {exc})")
            return True, ("stepped onto the leader's spot — its press-X "
                          "prompt is up, and a sigil admits only the "
                          "wizards standing on it")
        return False, ""              # already together, nothing to do

    # Onto the leader whenever they are fighting, however close this
    # wizard already is. `join_the_fight` steps into the duel by
    # teleporting to the CLOSEST mob, and closest is measured from
    # wherever the follower is standing -- so a follower inside the
    # radius but beside a different group walked into that group's fight
    # instead, and the party spent the duel in two duels. Landing on the
    # leader first makes "closest" mean the leader's circle.
    if gap is None or gap > radius or leader_fighting:
        try:
            await follower.teleport(target)
        except Exception as exc:
            return False, (f"could not teleport to the leader "
                           f"({type(exc).__name__}: {exc})")

    if leader_fighting:
        # The leader is mid-duel and this wizard is not in it. Standing
        # next to the circle contributes nothing -- and worse, it looks
        # exactly like a working party right up until the plan says
        # "one wizard in the fight".
        ok, why = await join_the_fight(follower)
        return (True, "joined the leader's fight") if ok else (False, why)

    return True, "regrouped on the leader"
