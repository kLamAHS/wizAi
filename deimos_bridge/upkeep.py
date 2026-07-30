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

#: `src/paths.py:16` — the potion button on the HUD. Its visibility is
#: the practical test for "the wizard is back in the world", and it is
#: also the exact window the potion click needs, so waiting for it is
#: waiting for the one thing both chores depend on.
HUD_POTION_PATH = ["WorldView", "windowHUD", "btnPotions"]

#: How far from a wisp still counts as having reached it. Wizard101's
#: world units put a pickup radius well inside this; the number only has
#: to separate "landed on it" from "snapped back to the duel circle".
ARRIVAL_RADIUS = 120.0


async def _read(obj, path, default=None):
    """Read `obj.a.b()`, or give up quietly.

    The attribute walk is inside the `try` on purpose: a wizwalker that
    renamed one of these, or a client stub that never had it, must read
    as "cannot tell" rather than raise out of a chore that is meant to
    be optional.
    """
    try:
        target = obj
        for part in path.split("."):
            target = getattr(target, part)
        return await target()
    except Exception:
        return default


async def _hud_visible(client):
    """True / False / None, where None means the question is unanswerable.

    None is not a failure. A client whose window tree cannot be walked at
    all (no `root_window`, a light stub, a wizwalker that renamed it)
    must not make the caller wait out a timeout for an answer that is
    never coming -- it should get on with the chores and say the gate was
    skipped.
    """
    from . import questing

    try:
        root = client.root_window
    except Exception:
        return None
    if root is None:
        return None
    try:
        window = await questing.window_from_path(root, HUD_POTION_PATH)
    except Exception:
        return None
    if window is None:
        return False
    return await questing._visible(window)


async def wait_until_free(client, timeout: float = 15.0, poll: float = 0.4,
                          settle: float = 0.8):
    """(ok, reason) — block until the wizard can actually be driven.

    **This is the fix for "collect wisps isn't working".** wizwalker
    defines `in_battle` as `duel_phase is not DuelPhase.ended`, and
    `CombatHandler.wait_for_combat` returns the instant that flips. At
    that moment the duel is over but the wizard is not free: the results
    screen is still up, the HUD is gone, and the body is still pinned in
    the duel circle. Both chores ran there.

    What that looked like from the outside: teleports were accepted and
    then snapped back, so the wisp sweep reported "collected 3 wisp(s)"
    having moved nobody; and the potion click went looking for
    `btnPotions`, which was not in the window tree yet, so it missed and
    reported no charges. Neither said a word about timing.

    Deimos gates the same two chores on `is_free` (`src/utils.py:776`).
    That alone is weaker than it looks here, because its `in_battle` term
    is the same predicate that has already flipped, so this waits for the
    HUD as well and then settles.

    The two halves are graded differently on purpose. Loading and combat
    are *hard* gates: driving through either is pointless, so a timeout
    there skips the chores. The HUD is a *soft* gate: if it never becomes
    visible -- or cannot be read at all, which is the light-install
    case -- the chores go ahead and the reason is reported, because doing
    nothing forever is the worse failure of the two.
    """
    waited, blocked = 0.0, ""
    while waited < timeout:
        loading = await _read(client, "is_loading", False)
        fighting = await _read(client, "in_battle", False)
        if not loading and not fighting:
            break
        blocked = "still loading" if loading else "still in the duel"
        await asyncio.sleep(poll)
        waited += poll
    else:
        return False, (f"{blocked} {timeout:.0f}s after the fight ended — "
                       "skipped the between-fights chores")

    waited_for_hud = False
    while waited < timeout:
        hud = await _hud_visible(client)
        if hud is None:
            return True, ""          # unanswerable; not worth a timeout
        if hud:
            break
        waited_for_hud = True
        await asyncio.sleep(poll)
        waited += poll
    else:
        return True, ("the HUD never came back after the fight — doing the "
                      "chores anyway, but a potion click may miss")

    if blocked or waited_for_hud:
        # Only when something was actually waited out. The settle is for
        # the frame or two between the HUD appearing and the body being
        # released; with nothing to wait for there is nothing to settle.
        await asyncio.sleep(settle)
    return True, ""


async def _arrived(client, target) -> bool:
    """Did the wizard actually end up at `target`?

    A teleport issued while the game still owns the body is accepted and
    then undone, and `client.teleport` raises nothing when that happens.
    Counting the call was therefore counting nothing. Returns True when
    the position cannot be read, so an unreadable client keeps the old
    behaviour rather than reporting every wisp as a miss.
    """
    here = await _read(client, "body.position", None)
    if here is None or target is None:
        return True
    try:
        dx = here.x - target.x
        dy = here.y - target.y
        dz = here.z - target.z
    except AttributeError:
        return True
    return (dx * dx + dy * dy + dz * dz) ** 0.5 <= ARRIVAL_RADIUS


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
        except Exception as exc:
            # The unfiltered list is still worth walking -- but say so.
            # Falling through silently turns `safe_only=True` into
            # `safe_only=False` without telling anyone, and the extra
            # duel that starts a minute later has nothing to attribute
            # it to. The checkbox promises this does not happen.
            say(f"could not check which wisps are safe "
                f"({type(exc).__name__}) — collecting all {found} of them, "
                f"which may pull a fight")
        if not entities:
            say(f"{found} wisp(s) found, all of them next to a mob")
            return 0

    collected, refused, snapped = 0, None, 0
    for entity in entities[:limit]:
        try:
            where = await entity.location()
            await client.teleport(where)
            await asyncio.sleep(0.15)
        except Exception as exc:
            refused = f"{type(exc).__name__}: {exc}"
            continue
        # Counting the *call* counted teleports the game had already
        # undone -- which is what "collected 3 wisp(s)" with no health
        # gained was. Count arrivals.
        if await _arrived(client, where):
            collected += 1
        else:
            snapped += 1
    if collected:
        say(f"collected {collected} wisp(s)"
            + (f" ({snapped} teleport(s) snapped back)" if snapped else ""))
    elif snapped:
        say(f"{snapped} wisp teleport(s) snapped back — the wizard never "
            f"moved, which is what happens while the game still owns it")
    elif refused:
        say(f"could not teleport to a wisp — {refused}")
    return collected


async def needs_potion(client, minimum_mana: int = 16) -> bool:
    """Deimos's threshold, reimplemented on plain stat reads.

    Low on mana, or under 55% health (`src/utils.py:527-544`).

    A failed read still answers False -- there is no safe way to spend a
    charge on a number nobody read -- but it now leaves the reason in
    `needs_potion.last_error`, because "healthy" and "unreadable" were
    the same answer and only one of them means potions are working. A
    wizard at 12% health with three charges in the bottle sat there all
    run, silently, because one of these five reads raised.
    """
    needs_potion.last_error = ""
    try:
        mana = await client.stats.current_mana()
        max_mana = await client.stats.max_mana()
        health = await client.stats.current_hitpoints()
        max_health = await client.stats.max_hitpoints()
        level = await client.stats.reference_level()
    except Exception as exc:
        needs_potion.last_error = (
            f"could not check whether you need a potion — "
            f"{type(exc).__name__}: {exc}")
        return False
    if not max_health:
        needs_potion.last_error = "your max health read back as 0"
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

    `last_error` is cleared on entry and set on every failing path. It
    used to be neither: a stale ModuleNotFoundError from the first fight
    of the run was still being reported eight fights later as the reason
    an empty bottle had not been drunk, and a *failed charge read* was
    reported as "no charges left" -- an assertion about the game that
    nothing had actually read. One of those sends you to a vendor to buy
    potions you already have.
    """
    drink_potion.last_error = ""
    try:
        if await client.stats.potion_charge() < 1.0:
            drink_potion.last_error = ("your potion bottle is empty — this "
                                       "never buys refills")
            return False
    except Exception as exc:
        drink_potion.last_error = (f"could not read your potion charges "
                                   f"({type(exc).__name__}: {exc})")
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
                      on_status=None, wait: bool = True):
    """The between-fights chores, in the order that wastes least.

    Wisps first: they are free, and topping up from them can put the
    wizard above the potion threshold so no charge is spent at all.

    `wait` gates everything on the wizard actually being free -- see
    `wait_until_free`, which is the reason this used to run against the
    battle-results screen and report success at doing nothing. It is a
    parameter only so tests can drive the chores directly.
    """
    def say(message):
        if on_status:
            on_status(message)

    if wait:
        free, why = await wait_until_free(client)
        if why:
            say(why)
        if not free:
            return

    if wisps:
        # `on_status` passed down: the count alone cannot distinguish
        # "there were none" from "the helper could not import".
        await collect_wisps(client, on_status=say)

    if not potions:
        return

    if not await needs_potion(client):
        why = getattr(needs_potion, "last_error", "")
        if why:
            # "did not need one" and "could not tell" are different
            # answers, and only the second is a problem to fix.
            say(why)
        return

    if await drink_potion(client):
        say("drank a potion")
    else:
        why = getattr(drink_potion, "last_error", "")
        say("low on health or mana, but no potion was drunk"
            + (f" — {why}" if why else ""))
