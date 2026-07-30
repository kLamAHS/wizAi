"""Deimos's real questing, driven one step at a time.

`questing.py` in this package walks to the quest marker and clicks
dialogue. That is enough to farm a fight you are standing next to and
not much more — it has no navigation, so a quest whose marker is across
a zone boundary stalls.

Deimos already solved that properly. `Deimos/src/questing.py` handles
navmap teleports, spiral doors, dungeon entry, NPC talking, zone
correction, chest rerolls and potions. Reimplementing any of it would be
worse than using it.

**Why this composes rather than conflicts.** `Quester.auto_quest_solo`
opens with `if await is_free(self.client):`, and `is_free` is False
during combat, a loading screen or dialogue (`src/questing.py:1414-1416`).
So it advances the quest when the wizard is idle and does nothing at all
once a duel starts. That is exactly the division this project needs:
Deimos gets the wizard to the fight, and wizAi's policy fights it. The
two never reach for the mouse at the same time.

Deimos's own loop is `while client.questing_status: sleep(1);
auto_quest_solo(...)` (`src/questing.py:1406-1409`). Rather than hand
over control, `step()` runs a single iteration, so the live worker keeps
ownership of the fight loop.

**What it costs.** `src.utils` imports
`wizwalker.extensions.wizsprinter.wiz_navigator`. That is *not* a PyPI
package -- it is vendored at `Deimos/libs/wizsprinter` and overlays into
the `wizwalker` namespace, so `deimos_path.ensure_path` puts it on the
path rather than anyone pip-installing it. What remains are ordinary
packages: `thefuzz`, `loguru`, `pyyaml`, `requests`, `lark`. Without them
`available()` names the missing one and the live worker falls back to the
light questing in `questing.py`; nothing else in the bridge is affected.
"""
import os

from .deimos_path import (DEIMOS_ROOT, ensure_path as _ensure_path,
                          install_hint)


def available():
    """(ok, reason). Can Deimos's questing be imported here?"""
    if not os.path.isdir(DEIMOS_ROOT):
        return False, f"no Deimos directory at {DEIMOS_ROOT}"
    _ensure_path()
    try:
        from src.questing import Quester  # noqa: F401
        return True, ""
    except Exception as exc:
        hint = install_hint(exc)
        return False, (
            f"Deimos's questing is not importable ({type(exc).__name__}: "
            f"{exc})."
            + (f"\n\n{hint}" if hint else "")
            + "\n\nUntil then the light questing in "
              "deimos_bridge/questing.py is used instead — it teleports "
              "and clicks dialogue, but cannot navigate across zones.")


#: The per-client attributes `Quester` reads. Deimos sets these in
#: `Deimos.py:_init_client_attrs`; wizwalker's Client has none of them,
#: so anything driving Quester outside Deimos has to supply them or the
#: first read raises AttributeError.
DEFAULTS = {
    "combat_status": False,
    "questing_status": True,      # the flag Deimos's own loop runs on
    "sigil_status": False,
    "auto_pet_status": False,     # no pet training on a data-collection run
    "feeding_pet_status": False,
    "use_team_up": False,
    "dance_hook_status": False,
    "entity_detect_combat_status": False,
    "invincible_combat_timer": False,
    "just_entered_combat": None,
    "just_left_combat": False,
    "helper_clients": None,       # replaced with [] below; lists are shared
    "client_being_helped": None,
    "original_location_before_combat": None,
    "duel_circle_joinable": True,
    "in_solo_zone": False,
    "wizard_name": None,
    "discard_duplicate_cards": False,
    "kill_minions_first": False,
    "automatic_team_based_combat": False,
    "latest_drops": "",
    "combat_config": "",
    "use_potions": True,          # staying alive matters when unattended
    "buy_potions": False,         # buying involves a vendor trip; opt in
    "client_to_follow": None,
    "player_gid": None,
}


async def init_client(client, **overrides):
    """Give a wizwalker Client the attributes Deimos's questing expects."""
    for name, value in DEFAULTS.items():
        setattr(client, name, [] if value is None and name == "helper_clients"
                else value)
    client.helper_clients = []
    for name, value in overrides.items():
        setattr(client, name, value)

    if not getattr(client, "title", None):
        client.title = "wizAi"
    try:
        client.character_level = await client.stats.reference_level()
    except Exception:
        client.character_level = 1
    return client


class DeimosQuester:
    """One `Quester`, stepped rather than looped."""

    def __init__(self, client, quester):
        self.client = client
        self.quester = quester
        self.failures = 0
        self.last_error = ""

    async def step(self, ignore_pet_level_up: bool = True,
                   play_dance_game: bool = False) -> bool:
        """Advance the quest by one iteration.

        Returns False if the step raised. Deimos's questing reads a lot
        of memory and a failed read mid-zone-change is routine, so a
        raise here is a skipped tick, not a broken run -- but the count
        is kept so a persistently failing setup is visible rather than
        silently doing nothing.
        """
        try:
            await self.quester.auto_quest_solo(
                auto_pet_disabled=True,
                ignore_pet_level_up=ignore_pet_level_up,
                play_dance_game=play_dance_game)
            self.failures = 0
            return True
        except Exception as exc:
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False


async def make_quester(client, **overrides):
    """Build a stepper for `client`, or raise with a readable reason."""
    ok, reason = available()
    if not ok:
        raise RuntimeError(reason)
    _ensure_path()
    from src.questing import Quester

    await init_client(client, **overrides)
    # Single-client questing is how Deimos itself drives it when there is
    # no leader: `Quester(client, walker.clients, None)` (Deimos.py:852).
    return DeimosQuester(client, Quester(client, [client], None))
