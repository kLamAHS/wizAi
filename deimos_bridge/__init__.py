"""Bridge between the wizAi combat research code and Deimos.

Deimos (`../Deimos`) is a Wizard101 automation bot built on wizwalker. It
reads the *live game client's* memory, so its combat model was derived by
watching real fights rather than by reading a wiki. That makes it two
things wizAi did not have before:

  * an **independent oracle** for the damage math, and
  * a way to actually cast wizAi's learned policy at real enemies.

The package is deliberately split so the first is usable on any machine
and only the second needs Windows and a running game:

  `deimos_damage`   pure-Python port of Deimos's `src/combat_math.py`.
                    No wizwalker, no async, no game. Runs anywhere.
  `scenarios`       one scenario spec rendered into both engines' terms,
                    so a comparison is apples-to-apples by construction.
  `differential`    runs both engines over the scenario suite and diffs.
  `effect_audit`    the real game's SpellEffects enum vs. what wizAi's
                    simulator actually implements.
  `live_state`      live wizwalker combat -> wizAi `State`.
  `live_backend`    a wizsprinter combat backend that asks a wizAi policy
                    what to cast. This is the piece that fights real
                    enemies.
  `mock_client`     fakes of the wizwalker objects, so `live_state` and
                    `live_backend` are testable without the game.

Import policy: nothing at this level imports wizwalker. The two live
modules import it lazily inside functions, so `import deimos_bridge` is
safe on Linux.
"""

__all__ = [
    "deimos_damage",
    "scenarios",
    "differential",
    "effect_audit",
    "live_state",
    "live_backend",
    "mock_client",
]
