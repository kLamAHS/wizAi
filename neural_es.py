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

    python3 neural_es.py GENERATIONS OUT.json [SEED] [SEED_WEIGHTS.json]
                         [BOARD_SET]
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

#: near the canonical contested set but never ON it (the held-out
#: verdict happens at hp 600/700/1300; fitness never does) -- and,
#: since the second campaign, WIDE: both mob counts per deck, the
#: buff-heavy deck the first campaign's winner measured worst on, and
#: a fourth school so the net cannot mistake ice-and-storm for the
#: whole game.
#:
#: CAMPAIGN 2 VERDICT -- rejected, recorded: the wide objective did
#: exactly what averaging does. Its best checkpoint (+2.6 wide-board
#: validation over the campaign-1 winner) fixed the buff-heavy board
#: completely (9.1 -> 23.9 held out) and PAID for it with both crowns:
#: storm 700x2 fell 29.9/31.5 -> 16.9/18.9 and low 700x2 22.4 -> 18.4.
#: Under per-deck routing that trade buys nothing -- on the buff-heavy
#: deck the sweep already installs school-aware(3) at 24.1, so 23.9
#: there is redundant, while storm is the one deck where the neural
#: candidate IS the pick. The shipped net therefore stays the
#: campaign-1 specialist. Lesson for campaign 3: with per-deck routing,
#: the one shipped net should specialise in what the hand-coded five
#: do NOT cover; a broad fitness set optimises toward redundancy. More
#: coverage means MORE seats (per-specialisation weights files), not a
#: wider objective on one.
_LOW = (["Frost Beetle"] * 4 + ["Ice Trap"] * 2 + ["Snow Serpent"] * 4
        + ["Evil Snowman"] * 4 + ["Tower Shield"] * 2)
_BUFFY = (["Iceblade"] * 4 + ["Ice Trap"] * 4 + ["Evil Snowman"] * 4
          + ["Frost Beetle"] * 4)
_STORM = (["Thunder Snake"] * 4 + ["Lightning Bats"] * 4
          + ["Storm Shark"] * 4 + ["Stormblade"] * 2)
_FIRE = (["Fire Cat"] * 4 + ["Fire Elf"] * 4 + ["Sunbird"] * 4
         + ["Fireblade"] * 2)
#: the defensive deck: shields and heals the hand-coded continuations
#: never cast inside a rollout
_MID = (["Colossus"] * 4 + ["Ice Wyvern"] * 4 + ["Iceblade"] * 4
        + ["Tower Shield"] * 3 + ["Sprite"] * 2)

#: An 8th element names a CATALOG boss: the board becomes the casting
#: encounter (pool, opening pips) at the probe health -- the opponent
#: the live rollouts actually price since _apply_pool.
BOARD_SETS = {
    "wide": (
        (_LOW, "ice", 1022, 0.09, 620, 2, 82),
        (_LOW, "ice", 1022, 0.09, 730, 2, 92),
        (_BUFFY, "ice", 1022, 0.09, 620, 2, 82),
        (_BUFFY, "ice", 1022, 0.09, 730, 2, 92),
        (_STORM, "storm", 800, 0.05, 620, 2, 82),
        (_STORM, "storm", 800, 0.05, 1250, 1, 105),
        (_FIRE, "fire", 900, 0.07, 620, 2, 82),
    ),
    # campaign 4: the two boards the spread survey measured as real
    # levers where a hand-coded candidate leads the neural one by ~10
    # -- both CHARM-HEAVY decks. buffy(ice): spread 34.4, leader
    # school-aware(0) 36.8 vs neural 26.8. L30 myth: spread 15.2,
    # leader school-aware(3) 43.6 vs neural 33.2. Near-ceiling boards
    # (L30 death, leader 96.8) are deliberately absent: matching a
    # covered leader is redundancy, the campaign-2 lesson. Fitness
    # healths sit off the survey points (650/950), which stay held
    # out; the bar to ship is beating the LEADER there, not the
    # shipped net.
    #
    # CAMPAIGN 4 VERDICT -- SHIPPED as the second seat (neural-b).
    # Run A climbed 31.8 -> 40.8 on fitness; held out on the survey
    # points it beat the buff-heavy board's leader on both fresh
    # streams (38.2/37.8 vs school-aware(0)'s 32.9/31.9) and closed
    # most of the myth gap without taking it (40.0 vs
    # school-aware(3)'s 41.1 -- that board stays hand-coded ground).
    # The survey-first design is what campaigns 2 and 3 bought: aim
    # at measured levers, ship on beating the leader, let the
    # per-deck sweep route everything else.
    "charm": (
        (["Iceblade"] * 4 + ["Ice Trap"] * 4 + ["Evil Snowman"] * 4
         + ["Frost Beetle"] * 4, "ice", 1022, 0.09, 620, 2, 92),
        (["Iceblade"] * 4 + ["Ice Trap"] * 4 + ["Evil Snowman"] * 4
         + ["Frost Beetle"] * 4, "ice", 1022, 0.09, 690, 2, 92),
        (["Cyclops"] * 4 + ["Humongofrog"] * 4 + ["Troll"] * 4
         + ["Balanceblade"] * 2 + ["Hex"] * 2 + ["Tower Shield"] * 2,
         "myth", 1270, 0.15, 900, 2, 115),
        (["Cyclops"] * 4 + ["Humongofrog"] * 4 + ["Troll"] * 4
         + ["Balanceblade"] * 2 + ["Hex"] * 2 + ["Tower Shield"] * 2,
         "myth", 1270, 0.15, 1010, 2, 115),
    ),
    # campaign 5: the survey pass-2 levers. With the two nets leading
    # 9 of 17 contested boards, the hand-coded-led remainder clusters
    # by MECHANIC: death at both tiers (Poison DoTs, Vampire/Wraith/
    # Scarecrow drains), myth at 2 mobs, life -- sustain decks, a
    # shape no net has trained on. Survey points (784/1097) stay held
    # out; the bars are the leaders there: school-aware(3) 30.4 on
    # L30 death, nuke-asap 60.8 on L50 death, school-aware(0) 58.8 on
    # L30 myth, school-aware(3) 72.8 on L30 life.
    #
    # CAMPAIGN 5 VERDICT -- SHIPPED as the third seat (neural-c). Run
    # A climbed 49.9 -> 55.0 on fitness; held out, it beat the L30
    # death leader on both fresh streams (34.9/38.1 vs school-aware(3)
    # 28.5/31.8), tied the L30 life leader exactly (69.0 vs 69.1),
    # and stayed behind the leaders on L50 death (56.0 vs nuke-asap
    # 62.9) and L30 myth (54.4 vs school-aware(0) 57.2) -- those
    # boards remain hand-coded ground, and the per-deck sweep keeps
    # them there.
    "sustain": (
        (["Poison"] * 4 + ["Vampire"] * 4 + ["Banshee"] * 4
         + ["Balanceblade"] * 2 + ["Hex"] * 2 + ["Death Shield"] * 2,
         "death", 1392, 0.21, 700, 2, 107),
        (["Poison"] * 4 + ["Vampire"] * 4 + ["Banshee"] * 4
         + ["Balanceblade"] * 2 + ["Hex"] * 2 + ["Death Shield"] * 2,
         "death", 1392, 0.21, 860, 2, 107),
        (["Scarecrow"] * 4 + ["Wraith"] * 4 + ["Poison"] * 4
         + ["Balanceblade"] * 2 + ["Hex"] * 2 + ["Death Shield"] * 2,
         "death", 1980, 0.25, 980, 2, 152),
        (["Scarecrow"] * 4 + ["Wraith"] * 4 + ["Poison"] * 4
         + ["Balanceblade"] * 2 + ["Hex"] * 2 + ["Death Shield"] * 2,
         "death", 1980, 0.25, 1210, 2, 152),
        (["Cyclops"] * 4 + ["Humongofrog"] * 4 + ["Troll"] * 4
         + ["Balanceblade"] * 2 + ["Hex"] * 2 + ["Tower Shield"] * 2,
         "myth", 1270, 0.21, 1000, 2, 97),
        (["Cyclops"] * 4 + ["Humongofrog"] * 4 + ["Troll"] * 4
         + ["Balanceblade"] * 2 + ["Hex"] * 2 + ["Tower Shield"] * 2,
         "myth", 1270, 0.21, 1200, 2, 97),
        (["Seraph"] * 4 + ["Nature's Wrath"] * 4 + ["Leprechaun"] * 4
         + ["Balanceblade"] * 2 + ["Hex"] * 2 + ["Tower Shield"] * 2,
         "life", 1503, 0.21, 1000, 2, 115),
        (["Seraph"] * 4 + ["Nature's Wrath"] * 4 + ["Leprechaun"] * 4
         + ["Balanceblade"] * 2 + ["Hex"] * 2 + ["Tower Shield"] * 2,
         "life", 1503, 0.21, 1200, 2, 115),
    ),
    # campaign 3: survival play against CASTING bosses -- the ground
    # no hand-coded continuation covers (none of the five ever casts a
    # shield or a heal inside a rollout) and the shipped specialist
    # was never trained on. Verdict boards use different healths
    # (Ketil 1300, Jack of Knaves 1100) and stay held out.
    #
    # CAMPAIGN 3 VERDICT -- rejected, recorded: the hypothesis was
    # wrong. The best checkpoint climbed +5.6 on validation and won
    # NOTHING held out (4th/5th/7th of seven candidates), because on
    # caster boards the continuation seat is nearly FLAT: all seven
    # candidates landed within 2.7 points on Ketil@1300 and within ~5
    # with stream-unstable ordering on Jack@1100. Those fights are
    # decided by the outer search and the spike arithmetic, not by
    # rollout continuation style -- there was no uncovered ground to
    # specialise into, and the validation gain was fitness-local.
    # Lesson for any campaign 4: hunt seats where the CANDIDATE SPREAD
    # is measured large first; absolute win rates that look improvable
    # say nothing about whether this seat is the lever that improves
    # them.
    "defense": (
        (_MID, "ice", 1800, 0.25, 1000, 1, 0, "Ketil Blackheart"),
        (_MID, "ice", 1800, 0.25, 1450, 1, 0, "Ketil Blackheart"),
        (_MID, "ice", 1800, 0.25, 950, 1, 0, "Jack of Knaves"),
        (_MID, "ice", 1800, 0.25, 4300, 1, 0, "Tomugawa the Evil"),
        (_LOW, "ice", 1022, 0.09, 640, 2, 88),
    ),
}
BOARDS = BOARD_SETS["wide"]

N_FIT = 30          # fights per board per candidate
N_VAL = 100         # fights per board for the validation score
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


def _board_sim(board):
    deck, school, php, dmgb = board[:4]
    hp, mobs, dmg = board[4:7]
    name = board[7] if len(board) > 7 else None
    if name:
        from deimos_bridge.bestiary import full_boss
        boss = full_boss(name, hp)
        extra = []
    else:
        boss = Boss(name="p", hp=hp, school="death", dmg=dmg)
        extra = [Boss(name=f"p{i}", hp=hp, school="death", dmg=dmg)
                 for i in range(1, mobs)]
    return Sim(CARDS, list(deck), school, boss, enemies=extra,
               player_hp=php, player_stats={"damage": {"*": dmgb}},
               rules=LIVE_RULES)


def fitness(theta, shapes, base_seed, n, boards=None):
    cont = _policy(_unflatten(theta, shapes))
    boards = BOARDS if boards is None else boards
    total = 0.0
    for board in boards:
        sim = _board_sim(board)
        pol = P.greedy_ttk(6, continuation=cont)
        wins = 0
        for i in range(n):
            sim.rng = random.Random(base_seed + i)
            _, won, _ = sim.run(pol, max_turns=25)
            wins += won
        total += wins / n
    return total / len(boards)


def main(generations, out_path, seed=0, seed_weights=None,
         board_set="wide"):
    rng = np.random.RandomState(seed)
    boards = BOARD_SETS[board_set]
    # resume a killed campaign from its own best checkpoint
    base = Net.load(seed_weights or DEFAULT_WEIGHTS)
    shapes = [(W.shape[0], W.shape[1]) for W, _ in base.layers]
    theta = _flatten(base)

    val0 = fitness(theta, shapes, 999_999, N_VAL, boards)
    best_val, best_theta = val0, theta.copy()
    print(f"gen 0 ({board_set} seed): val {val0 * 100:.1f}", flush=True)

    for gen in range(1, generations + 1):
        gen_seed = 1_000_000 + 10_000 * gen        # fresh per generation
        grad = np.zeros_like(theta)
        for _ in range(PAIRS):
            eps = rng.randn(len(theta))
            fp = fitness(theta + SIGMA * eps, shapes, gen_seed, N_FIT,
                         boards)
            fm = fitness(theta - SIGMA * eps, shapes, gen_seed, N_FIT,
                         boards)
            grad += (fp - fm) * eps
        theta = theta + ALPHA * grad / (2 * PAIRS * SIGMA)
        val = fitness(theta, shapes, 999_999, N_VAL, boards)
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
         int(sys.argv[3]) if len(sys.argv) > 3 else 0,
         sys.argv[4] if len(sys.argv) > 4 else None,
         sys.argv[5] if len(sys.argv) > 5 else "wide")
