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
    try:
        async with follower.mouse_handler:
            await teleport_to_friend_from_list(follower, name=leader_name)
    except Exception as exc:
        return False, (f"could not teleport to {leader_name} through the "
                       f"friends list ({type(exc).__name__}: {exc}) — they "
                       f"have to be on this wizard's friends list and online")
    return True, ""


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

    if gap is None or gap > radius:
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
