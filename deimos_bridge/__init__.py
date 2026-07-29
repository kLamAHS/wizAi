"""Run the wizAi learner against Deimos' combat model.

Deimos is a Wizard101 bot: it reads the live client's memory, so its
combat numbers are the game's numbers rather than a wiki's. That makes it
the natural adversary for a policy trained inside `w101_sim`, which only
ever saw scraped data.

Two ways to use it, in increasing order of how real the enemies are:

`deimos_bridge.engine`
    A headless duel loop driven by Deimos' own `sim_damage`. Runs on any
    machine, no game required, because Deimos' effect simulation turns
    out to be pure functions over plain dicts (see `headless`).

`deimos_bridge.live`
    A `CombatHandler` that plays actual fights on Windows with the game
    open, choosing cards with a trained wizAi policy.

`deimos_bridge.evaluate` scores a policy under both engines and reports
the gap, which is the number worth caring about: how much of the
learner's edge is real, and how much was an artifact of the simulator it
grew up in.
"""

from deimos_bridge.headless import install, deimos_modules

__all__ = ["install", "deimos_modules"]
