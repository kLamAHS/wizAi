"""A learned value for boards the rollout cannot play to an end.

`policies._rollout` scores a candidate by playing it out to a horizon.
When the horizon runs out — or the wizard dies inside it — the rollout
has nothing to say beyond a sentinel ranked by damage banked, and that
branch is not an edge case: it decides 17% of candidates on a level-5
board and 37–100% of them on a hard one. "Bank the most damage" is a
poor summary of a position; it cannot tell a board that is nearly won
from one that is nearly lost at the same damage total.

This module learns the summary instead: P(win) from a small vector of
**scale-free** features — fractions and ratios, no absolute health, no
damage number, no card identity. That shape is the entire point, and it
is chosen from measurement rather than taste:

  * The tabular key is 0% invariant to multiplying every health and
    damage by 20 — the one transformation levelling applies constantly —
    while a scale-free vector is bit-identical at any scale.
  * Card-identity features cost 46.2 points of coverage per renamed
    card; features of *behaviour* (how much damage is left in the deck,
    how many hits are in hand) survive any deck edit.

So one set of weights serves every deck, school, health pool and mob
count, which is what a game with 1,912 creatures requires. The weights
are linear and few — auditable, numpy-only, and cheap enough to price a
rollout leaf without slowing the search.

Training is self-play in the simulator: play randomized boards with a
fast scripted policy, record each planning phase's features, label every
state in a fight with its outcome, fit a logistic regression. Minutes,
once, for a function that never needs a health band.

STATUS: default OFF, twice measured. At horizon 12 it was a null --
only ~17% of decisions reach the stalled branch it ranks. The obvious
revisit once the tuner started installing horizon 6 (where far more
rollouts stall) was measured 2026-08: paired n=400 on the contested
set, +0.0/+0.2/+0.0 on three boards and **-15.8 on the buff-heavy
board**. A stalled H6 rollout with blades stacked is a nearly-won
position; nothing in these twelve features sees hanging charms, so the
leaf prices it like an ordinary attrition board and the search stops
building the very lines that win buff-heavy decks. Harmful, not null
-- do not enable at short horizons without a charm-aware feature set,
and measure in the seat before believing any refit.
"""
import json
import math
import os
import random

import numpy as np

#: Feature names, in vector order. The names are the audit: a weight on
#: "board_frac" is readable policy in a way a Q-table entry never is.
FEATURES = (
    "bias",
    "player_frac",        # my health / my max
    "board_frac",         # enemy health left / enemy health at start
    "alive_frac",         # living enemies / 4
    "weakest_frac",       # weakest living enemy / its max
    "kill_budget",        # deck+hand damage / enemy health left (capped)
    "death_clock",        # incoming per round * 4 / my health (capped)
    "pips_frac",          # effective pips / 14
    "hand_hits",          # damage cards in hand / 7
    "deck_hits",          # damage cards left anywhere / 8
    "blades_up",          # my positive charms / 3
    "focus_traps",        # wards on the weakest enemy / 3
)


def phi(sim, s):
    """The state as scale-free numbers. Cheap: called once per stalled
    rollout, not per turn."""
    p = s.player
    alive = [e for e in s.enemies if e.alive]
    board = sum(max(e.hp, 0.0) for e in alive)
    board_max = sum(max(e.max_hp, 1.0) for e in s.enemies) or 1.0
    incoming = sum(float(getattr(e, "flat_hit", 0.0) or 0.0) for e in alive)
    pool = list(s.hand) + list(s.deck)
    hits = [c for c in pool if c.kind in ("damage", "drain")]
    potential = sum(float(c.damage or 0.0) for c in hits)
    weakest = min(alive, key=lambda e: e.hp) if alive else None
    return np.array((
        1.0,
        max(0.0, p.hp) / max(p.max_hp, 1.0),
        board / board_max,
        len(alive) / 4.0,
        (max(0.0, weakest.hp) / max(weakest.max_hp, 1.0)) if weakest else 0.0,
        min(3.0, potential / max(board, 1.0)),
        min(3.0, incoming * 4.0 / max(p.hp, 1.0)),
        min(14, s.norm_pips + 2 * s.pow_pips) / 14.0,
        sum(1 for c in s.hand if c.kind in ("damage", "drain")) / 7.0,
        min(8, len(hits)) / 8.0,
        min(3, sum(1 for h in p.charms
                   if h.kind == "damage" and h.percent > 0)) / 3.0,
        (min(3, len(weakest.wards)) / 3.0) if weakest else 0.0,
    ))


class LeafValue:
    """P(win) from a state, as a callable the rollout can afford."""

    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=float)

    def __call__(self, sim, s):
        z = float(self.w @ phi(sim, s))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"features": list(FEATURES),
                       "weights": [float(v) for v in self.w]}, f, indent=1)

    @classmethod
    def load(cls, path=None):
        path = path or default_path()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if list(data.get("features") or []) != list(FEATURES):
            raise ValueError(
                "leaf_value.json was fitted for different features — retrain "
                "rather than silently mis-indexing the weights")
        return cls(data["weights"])


def default_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "leaf_value.json")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def _rand_board(rng, schools):
    def log_u(lo, hi):
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))

    n = rng.choice((1, 1, 2, 2, 3, 4))
    return ([int(log_u(80, 2500)) for _ in range(n)],
            [rng.choice(schools) for _ in range(n)],
            int(log_u(20, 160)), int(log_u(400, 4000)))


def collect(cards, decks, episodes=4000, seed=0, on_progress=None):
    """(X, y) from self-play on randomized boards.

    `decks` is [(decklist, school)]. Every planning phase contributes one
    row labelled with the fight's outcome — states from a won fight are
    1, from a lost one 0, which is exactly the quantity the rollout needs
    at a leaf.
    """
    from w101_sim import Boss, Sim
    from .policies import school_aware_blade_stack

    rng = random.Random(seed)
    mob_schools = ("fire", "ice", "storm", "myth", "life", "death", "balance")
    X, y = [], []
    for ep in range(episodes):
        deck, school = decks[rng.randrange(len(decks))]
        hps, mschools, dmg, php = _rand_board(rng, mob_schools)
        boss = Boss(name="b", hp=hps[0], school=mschools[0], dmg=dmg)
        extra = [Boss(name=f"m{i}", hp=h, school=ms, dmg=dmg)
                 for i, (h, ms) in enumerate(zip(hps[1:], mschools[1:]), 1)]
        sim = Sim(cards, deck, school, boss, player_hp=php,
                  rng=random.Random(rng.getrandbits(32)), enemies=extra)
        policy = school_aware_blade_stack(3)
        s = sim.new_state()
        rows = []
        won = False
        for _turn in range(30):
            rows.append(phi(sim, s))
            move = policy(sim, s)
            if move is not None:
                card, target = (move if isinstance(move, tuple)
                                else (move, 0))
                try:
                    sim.cast(s, card, target)
                except Exception:
                    pass
            if not any(e.alive for e in s.enemies):
                won = True
                break
            sim.end_round(s)
            if s.player_hp <= 0:
                break
            if not any(e.alive for e in s.enemies):
                won = True
                break
        X.extend(rows)
        y.extend([1.0 if won else 0.0] * len(rows))
        if on_progress and (ep + 1) % 500 == 0:
            on_progress(ep + 1, episodes)
    return np.array(X), np.array(y)


def fit(X, y, epochs=400, lr=0.5, l2=1e-4):
    """Plain logistic regression, full batch. Linear on purpose: the
    earlier measurement had the linear head matching or beating a small
    MLP on five of six transfer splits, and 12 auditable weights beat 385
    unauditable ones at the same score."""
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(epochs):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w -= lr * ((X.T @ (p - y)) / n + l2 * w)
    return LeafValue(w)


def auc(model, X, y):
    """Rank quality on held-out states; the transfer check's metric."""
    z = X @ model.w
    order = np.argsort(z)
    ranks = np.empty(len(z))
    ranks[order] = np.arange(1, len(z) + 1)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    if not n1 or not n0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
