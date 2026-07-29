"""wizAi's duel loop, with Deimos' arithmetic where the two disagree.

The point of this module is to change exactly one thing at a time.

`w101_sim` and Deimos model the same fights but not the same wizard. The
largest difference is not a rule either engine is missing -- it is that
wizAi treats gear damage as linear all the way up, while the client bends
it: past a soft cap each further point of damage buys less than the last,
and past a hard cap nothing at all. `w101_sim._damage_bonus` is
`1 + damage['fire'] + damage['*']` and stops there. Deimos runs the same
number through `curve_stat` first.

At level 20 the gear layer offers about 5% damage and the curve is a
no-op, so both engines agree. At level 120 it offers 147%, which is above
any plausible cap, so they cannot. A policy that learned to stack damage
under the linear model has been trained in a world where the last blade
is worth as much as the first.

`DeimosSim` is `Sim` with `curve_stat` -- Deimos' own function, imported,
not reimplemented -- spliced into the two places it belongs, and Deimos'
crit formula behind wizAi's crit hook. Everything else (cards, pips,
draws, enemy AI, cheats, wards) stays wizAi's, because those are not
what is in dispute and changing them would confound the measurement.

    >>> from deimos_bridge.engine import DeimosSim
    >>> sim = DeimosSim(cards, deck, "fire", boss, **loadout(120, "fire"))
    >>> sim.run(policy)                      # doctest: +SKIP
"""

from __future__ import annotations

import random

from w101_sim import Sim, _live_aura

from deimos_bridge.caches import DuelCurve
from deimos_bridge.headless import install

_MODS = install()
_curve_stat = _MODS.combat_math.curve_stat
_calc_crit = _MODS.effect_simulation.calc_crit


def deimos_crit_resolver(curve: DuelCurve, player_level: int = 100,
                         enemy_level: int = 100):
    """A wizAi `Rules.crit_resolver` backed by Deimos' `calc_crit`.

    Deimos answers "will this probably crit" with a threshold test,
    because it is deciding what a bot should expect. A simulator wants
    the distribution instead, so we take Deimos' chance and multiplier
    and roll against them.

    `crit_chance` on an Actor is read as a **rating** here, matching
    wizAi's own `make_rating_crit` convention: a resolver is installed
    precisely so those fields stop meaning probabilities.
    """

    def resolve(attacker, defender, rng, rules):
        crit_rating = max(attacker.crit_chance, 0.0)
        block_rating = max(defender.block_chance, 0.0)
        caster_level = player_level if attacker.team == 0 else enemy_level
        target_level = enemy_level if attacker.team == 0 else player_level

        multiplier, chance, block_chance = _calc_crit(
            crit_rating, block_rating, caster_level, target_level,
            curve.pvp or curve.raid)

        if chance <= 0 or rng.random() >= chance:
            return 1.0
        if block_chance > 0 and rng.random() < block_chance:
            return 1.0
        return multiplier

    return resolve


class DeimosSim(Sim):
    """`Sim`, with the client's stat curves in place of wizAi's linear ones.

    Accepts everything `Sim` does, plus `curve` (a `DuelCurve`) and
    `curve_enemies`. The client curves players only -- an enemy's damage
    and resist are used raw -- and we keep that asymmetry by default,
    since it is the behaviour Deimos implements.
    """

    def __init__(self, *args, curve: DuelCurve | None = None,
                 curve_enemies: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.curve = curve or DuelCurve()
        self.curve_enemies = curve_enemies

    def _curved(self, actor) -> bool:
        return self.curve_enemies or actor.team == 0

    def _damage_bonus(self, caster, school):
        # `Sim` returns 1 + total; the curve applies to the total.
        total = super()._damage_bonus(caster, school) - 1.0
        if self._curved(caster) and total > 0:
            total = _curve_stat(total, self.curve.damage_limit,
                                self.curve.d_k0, self.curve.d_n0)
        return 1.0 + total

    def _resist_mult(self, target, school, pierce):
        # Mirrors `Sim._resist_mult`, with the curve at Deimos' position:
        # resist is bent first, and pierce is subtracted from the bent
        # value. Doing it the other way round would let pierce eat the
        # part of resist the cap had already removed.
        res = target.resist.get(school, 0.0) + target.resist.get("*", 0.0)
        aura = _live_aura(target)
        if aura and (aura.schools is None or school in aura.schools):
            res -= aura.incoming
        if self._curved(target) and res > 0:
            res = _curve_stat(res, self.curve.resist_limit,
                              self.curve.r_k0, self.curve.r_n0)
        res = max(res - pierce, 0.0) if res > 0 else res
        return (1 - res) * (1 + target.boost.get(school, 0.0))


def paired_engines(cards, decklist, school, boss, *, curve=None,
                   seed=10 ** 6, **kwargs):
    """One `Sim` and one `DeimosSim` over identical inputs.

    Same cards, same deck, same boss, same stats -- so any difference in
    what comes out is the stat model and nothing else. Seed both from the
    same stream before each run to compare on matched draws.
    """
    base = Sim(cards, decklist, school, boss, **kwargs)
    curved = DeimosSim(cards, decklist, school, boss, curve=curve, **kwargs)
    base.rng = random.Random(seed)
    curved.rng = random.Random(seed)
    return base, curved
