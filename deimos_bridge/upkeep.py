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
    from .deimos_path import ensure_path

    # `ensure_path`, not a hand-rolled sys.path.insert. It also overlays
    # the vendored wizsprinter into the wizwalker namespace, which
    # `src.utils` (and therefore the potion helper) needs -- and it does
    # so by extending `wizwalker.extensions.__path__` rather than by
    # putting a directory on sys.path, which would make `wizwalker`
    # resolvable as a namespace package and shadow the real one.
    ensure_path()
    from src.sprinty_client import SprintyClient
    return SprintyClient(client)


async def available():
    """(ok, reason) — can the upkeep helpers run here?"""
    try:
        _sprinty(None)
        return True, ""
    except Exception as exc:
        from .deimos_path import install_hint
        hint = install_hint(exc)
        return False, (f"Deimos's SprintyClient is not importable "
                       f"({type(exc).__name__}: {exc})"
                       + (f". {hint}" if hint else ""))


async def collect_wisps(client, safe_only: bool = True, limit: int = 12,
                        on_status=None):
    """Teleport over the health/mana wisps lying around.

    `safe_only` keeps to wisps that are not sitting next to a mob, using
    Deimos's own `find_safe_entities_from` — walking into a second fight
    while topping up is a bad trade on a data run.

    Returns how many were collected. Bounded by `limit` so a zone strewn
    with pickups cannot stall the loop.

    **Every failure here used to be silent.** Five separate `except`
    blocks each returned or continued without a word, so a wizard whose
    wisps were never collected got no message, no log line, and no way to
    tell "there were none" apart from "the import failed" or "the
    teleport was refused". That is what "collect wisps isn't working"
    looks like from the outside. `on_status` is how it says which.
    """
    def say(message):
        if on_status:
            on_status(message)

    try:
        sprinty = _sprinty(client)
    except Exception as exc:
        say(f"wisps unavailable — {type(exc).__name__}: {exc}")
        return 0

    entities, read_failed = [], []
    for name in WISP_NAMES:
        try:
            entities += await sprinty.get_base_entities_with_vague_name(name)
        except Exception as exc:
            read_failed.append(f"{name} ({type(exc).__name__})")
    if not entities:
        say("no wisps in this zone"
            if not read_failed
            else "could not read the wisp entities: " + ", ".join(read_failed))
        return 0

    found = len(entities)
    if safe_only:
        try:
            entities = await sprinty.find_safe_entities_from(entities)
        except Exception:
            pass          # the unfiltered list is still worth walking
        if not entities:
            say(f"{found} wisp(s) found, all of them next to a mob")
            return 0

    collected, refused = 0, None
    for entity in entities[:limit]:
        try:
            await client.teleport(await entity.location())
            await asyncio.sleep(0.15)
            collected += 1
        except Exception as exc:
            refused = f"{type(exc).__name__}: {exc}"
    if collected:
        say(f"collected {collected} wisp(s)")
    elif refused:
        say(f"could not teleport to a wisp — {refused}")
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
        # `ensure_path` rather than a bare sys.path insert: `src.utils`
        # imports `wizwalker.extensions.wizsprinter.wiz_navigator`, and
        # the overlay that makes the vendored wizsprinter importable
        # lives there. Without it this raised ModuleNotFoundError and
        # returned False, which read as "no potion charge".
        from .deimos_path import ensure_path
        ensure_path()
        from src.utils import use_potion
        await use_potion(client)
        return True
    except Exception as exc:
        drink_potion.last_error = f"{type(exc).__name__}: {exc}"
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
        # `on_status` passed down: the count alone cannot distinguish
        # "there were none" from "the helper could not import".
        await collect_wisps(client, on_status=say)

    if potions and await needs_potion(client):
        if await drink_potion(client):
            say("drank a potion")
        else:
            why = getattr(drink_potion, "last_error", "")
            say("low on health or mana, but no potion was drunk"
                + (f" — {why}" if why else " — no charges left"))
