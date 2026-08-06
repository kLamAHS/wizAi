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


#: resolved full friends-list names, keyed by the short name a duel
#: gives. One lookup per leader per run rather than one per follow.
_FULL_NAMES = {}


async def friends_list_name(follower, short_name):
    """The leader's FULL name as the friends list spells it, or "".

    A combat read gives a wizard's first name -- "Jeffrey" -- and the
    friends list holds "Jeffrey IslandBringer". wizwalker matches on
    `friend_name == name`, exactly, so the teleport could never find a
    leader it was looking at: "Could not find friend with icon None icon
    list None and/or name Jeffrey", forever, while the friends window
    sat open next to it showing exactly one online friend.

    So the list is read and the entry whose name starts with what a duel
    told us is taken. Prefix rather than fuzzy: first names are what the
    duel reports and what the friends list leads with, and anything
    looser would happily teleport to the wrong friend.
    """
    if short_name in _FULL_NAMES:
        return _FULL_NAMES[short_name]
    try:
        from .deimos_path import ensure_path

        ensure_path()
        from wizwalker.extensions.scripting.utils import (
            _friend_list_entry, _maybe_get_named_window)
    except Exception:
        return ""

    try:
        root = follower.root_window
        try:
            window = await _maybe_get_named_window(root, "NewFriendsListWindow")
        except ValueError:
            button = await _maybe_get_named_window(root, "btnFriends")
            await follower.mouse_handler.click_window(button)
            window = await _maybe_get_named_window(root, "NewFriendsListWindow")
        listing = await _maybe_get_named_window(window, "listFriends")
        text = await listing.maybe_text() or ""
    except Exception:
        return ""

    for entry in _friend_list_entry.finditer(text):
        found = entry.group("name")
        if found == short_name or found.startswith(short_name + " "):
            _FULL_NAMES[short_name] = found
            return found
    return ""


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
        from wizwalker.extensions.scripting import teleport_to_friend_from_list
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
            return False, (
                f"the friends-list teleport to {name} ran for "
                f"{TELEPORT_TIMEOUT:.0f}s without finishing and was cut "
                f"off. It gets stuck paging the list to 'Online Friends' "
                f"when that tab will not come up — check the friends "
                f"window is not already open on another tab, or turn "
                f"'follow the leader' off and walk this wizard over")
        except ValueError as exc:
            if "Could not find friend" in str(exc):
                return False, ""
            return False, (f"could not teleport to {name} through the "
                           f"friends list (ValueError: {exc})")
        except Exception as exc:
            return False, (f"could not teleport to {name} through the "
                           f"friends list ({type(exc).__name__}: {exc})")
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
