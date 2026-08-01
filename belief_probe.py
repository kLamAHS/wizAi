"""Belief-vs-truth A/B: what is a correct opponent model worth, live?

The live read models every enemy as a flat measured hit -- the mean
health lost per round, split across the living mobs. That gets the
AVERAGE right and the SHAPE wrong: a casting boss saving six pips deals
nothing for three rounds and then a third of the wizard's health at
once, and shield-or-race is decided exactly by that shape. The client
reports every member's pip rack, and the catalog knows what 1,795
creatures cast -- so the live rollouts CAN model the boss as a caster
with known pips (`live_backend._apply_pool`). This probe measures what
that buys before believing it.

Design: the truth environment is a Sim with the full catalog casting
boss. Both arms play the same truth fights on paired seeds; the ONLY
difference is the state handed to the policy each round. The `pool` arm
sees the truth state (spell pool + true pips -- what `_apply_pool` plus
the pips read now provides, with an exact pool). The `flat` arm sees a
degraded clone reproducing the pre-change live read: pool stripped,
enemy pips zeroed, flat_hit set by the same measured-mean-with-hp/12-
prior rule as `_estimate_incoming`.

MEASURED (greedy_ttk, N=250 paired seeds, contested boards only --
ceiling boards rank nothing, same lesson as the envelope probes):

    Ketil Blackheart (storm, 4-pip opener), hp 900/1100/1300/1500:
        pool +10.0 / +2.8 / +8.0 / +9.2 points of kill rate
    Usunoki (fire hitter) hp 5760:            pool +8.4
    Vassanji Lore Singer (myth) hp 4800:      pool -0.8   (noise)
    Tomugawa the Evil (death, DoT) hp 4680:   pool -2.0   (noise)

Mean +5.1 over seven boards; every gain 3-4x the ~2.4-point seed-noise
floor, both losses inside it. The gains concentrate where they should:
hitter pools whose spikes decide the fight. CAVEAT recorded with the
result: in this harness the pool arm's opponent model is EXACT, because
belief pool == truth pool. Live, the catalog pool is approximate, so
these numbers are upper bounds of the live gain -- the reason
`_apply_pool` keeps the measured flat hit on every enemy as the
fallback the sim uses wherever a pool is absent or unresolvable.
"""
import random
import sys

from data_full import LIVE_RULES, load_spells_full
from deimos_bridge import bestiary
from deimos_bridge.policies import greedy_ttk
from search_policy import clone_state
from w101_sim import Sim


class TrueBelief:
    """Symmetry arm: clone, hand over untouched (pool + true pips)."""

    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        pass

    def __call__(self, sim, s):
        return self.policy(sim, clone_state(s))


class FlatBelief:
    """The pre-change live read: pool unknown, pips unread, damage =
    measured mean hp-loss per living enemy with the hp/12 prior
    (mirrors `live_backend._estimate_incoming`)."""

    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        self._hp = None
        self._n = 1
        self._obs = []

    def __call__(self, sim, s):
        p = s.player
        if self._hp is not None and p.hp < self._hp:
            self._obs.append((self._hp - p.hp) / max(1, self._n))
        self._hp = p.hp
        self._n = max(1, sum(1 for e in s.enemies if e.alive))
        per = (sum(self._obs) / len(self._obs)) if self._obs \
            else max(30.0, p.max_hp / 12)
        b = clone_state(s)
        for e in b.enemies:
            e.spell_pool = None
            e.norm_pips = e.pow_pips = e.school_pips = 0
            e.flat_hit = per
        return self.policy(sim, b)


DECK = (["Colossus"] * 4 + ["Ice Wyvern"] * 4 + ["Iceblade"] * 4
        + ["Tower Shield"] * 3 + ["Sprite"] * 2)
PLAYER = dict(player_hp=1800, player_stats={"damage": {"*": 0.25},
                                            "accuracy": 0.08})

#: (catalog name, observed hp) -- the contested points found by the
#: hp sweeps; catalog-health boards for these decks sit at 98-100%
#: where no belief can rank anything.
BOARDS = [
    ("Ketil Blackheart", 900), ("Ketil Blackheart", 1100),
    ("Ketil Blackheart", 1300), ("Ketil Blackheart", 1500),
    ("Usunoki", 5760),
    ("Vassanji Lore Singer", 4800),
    ("Tomugawa the Evil", 4680),
]


def duel(cards, boss, wrap, n, cfg=PLAYER, deck=DECK):
    wins = 0
    for i in range(n):
        pol = wrap(greedy_ttk())
        pol.reset()
        sim = Sim(cards, list(deck), "ice", boss,
                  player_hp=cfg["player_hp"],
                  player_stats=cfg["player_stats"],
                  rules=LIVE_RULES, rng=random.Random(10_000 + i))
        _, won, _ = sim.run(pol, max_turns=25)
        wins += won
    return wins / n * 100


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    cards = load_spells_full()
    for name, hp in BOARDS:
        boss = bestiary.full_boss(name, hp)
        pool = duel(cards, boss, TrueBelief, n)
        flat = duel(cards, boss, FlatBelief, n)
        print(f"{name[:22]:22s} hp {hp:5d}: pool {pool:5.1f}%  "
              f"flat {flat:5.1f}%  d={pool - flat:+.1f}", flush=True)
