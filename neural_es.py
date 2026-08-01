"""Optimizing the continuation seat directly -- no torch, no teacher.

The v2 featurizer experiment recorded the lesson this script acts on:
mimicry accuracy and playing strength are different objectives, and
behaviour cloning can only ever climb the first. Here the objective IS
the seat: fitness of a weight vector is the paired win rate of
greedy_ttk(6, continuation=net) on contested boards, and evolution
strategies climb that, seeded from the shipped BC weights.

Why ES fits this exactly:
- The net is ~6.5k parameters and DETERMINISTIC, so a small weight
  step flips a handful of discrete decisions -- and common random
  numbers (every candidate in a generation plays the same fight
  seeds) make those few flips visible over the fight noise.
- Antithetic pairs (theta +/- sigma*eps share seeds) cancel most of
  what CRN leaves.
- Seeds go FRESH each generation, so the optimizer cannot memorise a
  stream; and the running mean is scored each generation on a fixed
  VALIDATION stream it never trains on, with the best checkpoint kept.
  The worst case is therefore a recorded null, never a regression.

The final verdict is not this script's own numbers: the best
checkpoint faces the shipped v1 weights on the canonical held-out
boards at n=800 paired, fresh streams, and must keep the storm win
that earned the seat.

    python3 neural_es.py GENERATIONS OUT.json [SEED]
"""
import json
import random
import sys

import numpy as np

sys.path.insert(0, ".")

from data_full import LIVE_RULES, load_spells_full
from deimos_bridge import policies as P
from deimos_bridge.neural_net import DEFAULT_WEIGHTS, Net
from w101_sim import Boss, Sim

CARDS = load_spells_full()

#: near the canonical contested set but not identical to it -- the
#: held-out verdict happens on the canonical boards, not these
BOARDS = (
    (["Frost Beetle"] * 4 + ["Ice Trap"] * 2 + ["Snow Serpent"] * 4
     + ["Evil Snowman"] * 4 + ["Tower Shield"] * 2,
     "ice", 1022, 0.09, 650, 2, 85),
    (["Iceblade"] * 4 + ["Ice Trap"] * 4 + ["Evil Snowman"] * 4
     + ["Frost Beetle"] * 4, "ice", 1022, 0.09, 650, 2, 85),
    (["Thunder Snake"] * 4 + ["Lightning Bats"] * 4 + ["Storm Shark"] * 4
     + ["Stormblade"] * 2, "storm", 800, 0.05, 650, 2, 85),
    (["Thunder Snake"] * 4 + ["Lightning Bats"] * 4 + ["Storm Shark"] * 4
     + ["Stormblade"] * 2, "storm", 800, 0.05, 1300, 1, 110),
)

N_FIT = 40          # fights per board per candidate
N_VAL = 120         # fights per board for the validation score
SIGMA = 0.05
ALPHA = 0.03
PAIRS = 8           # antithetic pairs per generation


def _flatten(net):
    return np.concatenate([np.concatenate([W.ravel(), b.ravel()])
                           for W, b in net.layers])


def _unflatten(theta, shapes):
    layers, off = [], 0
    for (wi, wo) in shapes:
        W = theta[off:off + wi * wo].reshape(wi, wo); off += wi * wo
        b = theta[off:off + wo]; off += wo
        layers.append((W, b))
    return Net(layers)


def _policy(net):
    from deimos_bridge.neural_net import decision_matrix

    def strat(sim, s):
        cands, X = decision_matrix(sim, s)
        card, t = cands[int(np.argmax(net.scores(X)))]
        return None if card is None else (card, t)

    return strat


def fitness(theta, shapes, base_seed, n):
    cont = _policy(_unflatten(theta, shapes))
    total = 0.0
    for deck, school, php, dmgb, hp, mobs, dmg in BOARDS:
        boss = Boss(name="p", hp=hp, school="death", dmg=dmg)
        extra = [Boss(name=f"p{i}", hp=hp, school="death", dmg=dmg)
                 for i in range(1, mobs)]
        sim = Sim(CARDS, list(deck), school, boss, enemies=extra,
                  player_hp=php, player_stats={"damage": {"*": dmgb}},
                  rules=LIVE_RULES)
        pol = P.greedy_ttk(6, continuation=cont)
        wins = 0
        for i in range(n):
            sim.rng = random.Random(base_seed + i)
            _, won, _ = sim.run(pol, max_turns=25)
            wins += won
        total += wins / n
    return total / len(BOARDS)


def main(generations, out_path, seed=0):
    rng = np.random.RandomState(seed)
    base = Net.load(DEFAULT_WEIGHTS)
    shapes = [(W.shape[0], W.shape[1]) for W, _ in base.layers]
    theta = _flatten(base)

    val0 = fitness(theta, shapes, 999_999, N_VAL)
    best_val, best_theta = val0, theta.copy()
    print(f"gen 0 (BC seed): val {val0 * 100:.1f}", flush=True)

    for gen in range(1, generations + 1):
        gen_seed = 1_000_000 + 10_000 * gen        # fresh per generation
        grad = np.zeros_like(theta)
        for _ in range(PAIRS):
            eps = rng.randn(len(theta))
            fp = fitness(theta + SIGMA * eps, shapes, gen_seed, N_FIT)
            fm = fitness(theta - SIGMA * eps, shapes, gen_seed, N_FIT)
            grad += (fp - fm) * eps
        theta = theta + ALPHA * grad / (2 * PAIRS * SIGMA)
        val = fitness(theta, shapes, 999_999, N_VAL)
        mark = ""
        if val > best_val:
            best_val, best_theta = val, theta.copy()
            _unflatten(best_theta, shapes).save(out_path)
            mark = "  <- checkpoint"
        print(f"gen {gen}: val {val * 100:.1f} "
              f"(best {best_val * 100:.1f}){mark}", flush=True)

    if best_val <= val0:
        print(f"NULL: never beat the BC seed on validation "
              f"({best_val * 100:.1f} vs {val0 * 100:.1f})", flush=True)
    else:
        print(f"best checkpoint {best_val * 100:.1f} vs seed "
              f"{val0 * 100:.1f} -> held-out verdict next", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 0)
