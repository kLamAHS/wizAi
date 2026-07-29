"""Keeping the wizard alive between fights.

An unattended run dies by attrition long before it runs out of quests,
and a policy that lost because the wizard was at 12% health has not told
you anything about the policy. Two chores fix most of that: walk over the
wisps a fight drops, and drink a potion when low.

Both come from Deimos, but only through `SprintyClient` — which is
**pure wizwalker** (`src/sprinty_client.py` imports nothing else). That
matters: `collect_wisps` as Deimos ships it lives in `src/utils.py`,
which drags in `wizwalker.extensions.wizsprinter` and with it the Python
3.13 floor. Rebuilding the same three calls on `SprintyClient` directly
means upkeep works on the light install too, with no extra dependency.

Deimos's own questing already does this inside `auto_quest_solo`, but
only when `is_potion_needed` says so, and only while questing. Here it is
a toggle of its own so it also runs when auto-questing is off — which is
the case when farming one fixed mob, exactly where attrition bites.
"""
import asyncio

#: Entity name fragments the game uses for pickups. `WispGold` is
#: included because it is free and on the way; nothing here walks out of
#: its path for it.
WISP_NAMES = ("WispHealth", "WispMana", "WispGold")


def _sprinty(client):
    """Deimos's `SprintyClient`, which needs only wizwalker."""
    import os
    import sys

    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "Deimos")
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.sprinty_client import SprintyClient
    return SprintyClient(client)


async def available():
    """(ok, reason) — can the upkeep helpers run here?"""
    try:
        _sprinty(None)
        return True, ""
    except Exception as exc:
        return False, (f"Deimos's SprintyClient is not importable "
                       f"({type(exc).__name__}: {exc})")


async def collect_wisps(client, safe_only: bool = True, limit: int = 12):
    """Teleport over the health/mana wisps lying around.

    `safe_only` keeps to wisps that are not sitting next to a mob, using
    Deimos's own `find_safe_entities_from` — walking into a second fight
    while topping up is a bad trade on a data run.

    Returns how many were collected. Bounded by `limit` so a zone strewn
    with pickups cannot stall the loop.
    """
    try:
        sprinty = _sprinty(client)
    except Exception:
        return 0

    entities = []
    for name in WISP_NAMES:
        try:
            entities += await sprinty.get_base_entities_with_vague_name(name)
        except Exception:
            continue
    if not entities:
        return 0

    if safe_only:
        try:
            entities = await sprinty.find_safe_entities_from(entities)
        except Exception:
            pass

    collected = 0
    for entity in entities[:limit]:
        try:
            await client.teleport(await entity.location())
            await asyncio.sleep(0.15)
            collected += 1
        except Exception:
            continue
    return collected


async def needs_potion(client, minimum_mana: int = 16) -> bool:
    """Deimos's threshold, reimplemented on plain stat reads.

    Low on mana, or under 55% health (`src/utils.py:527-544`).
    """
    try:
        mana = await client.stats.current_mana()
        max_mana = await client.stats.max_mana()
        health = await client.stats.current_hitpoints()
        max_health = await client.stats.max_hitpoints()
        level = await client.stats.reference_level()
    except Exception:
        return False
    if not max_health:
        return False
    if minimum_mana > level:
        minimum_mana = level
    combined = int(0.23 * max_mana) + minimum_mana
    return mana < combined or float(health) / float(max_health) < 0.55


async def drink_potion(client) -> bool:
    """Use one potion if the wizard has a charge.

    Deliberately never *buys*: refilling means a trip to a vendor, real
    gold, and a navigation detour that can strand the run somewhere the
    quest is not. Deimos will buy if you ask it to; this will not.
    """
    try:
        if await client.stats.potion_charge() < 1.0:
            return False
    except Exception:
        return False
    try:
        import os
        import sys
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "Deimos")
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.utils import use_potion
        await use_potion(client)
        return True
    except Exception:
        # use_potion lives in src.utils, which needs wizsprinter. Without
        # it wisps still work; potions just do not.
        return False


async def after_fight(client, wisps: bool = True, potions: bool = True,
                      on_status=None):
    """The between-fights chores, in the order that wastes least.

    Wisps first: they are free, and topping up from them can put the
    wizard above the potion threshold so no charge is spent at all.
    """
    def say(message):
        if on_status:
            on_status(message)

    if wisps:
        n = await collect_wisps(client)
        if n:
            say(f"collected {n} wisp(s)")

    if potions and await needs_potion(client):
        if await drink_potion(client):
            say("drank a potion")
        else:
            say("low on health or mana, and no potion charge left")
