"""Should a wizard's lookahead know that its party is hitting too?

`live_state.read_state` already puts the other party members into
`State.allies` -- every non-hostile participant lands there -- and the
simulator already acts them out: `Sim.end_round` calls `_minion_turn` on
every living ally, which strikes the first living enemy for
`minion_power`. That field is **0.0** for a read wizard, so today every
rollout in a four-wizard fight models the other three as contributing
nothing for the whole horizon.

Filling the field in is a two-line change. This asks whether it is worth
making, because it is not obviously free: knowing the board will die
sooner makes setup worth *less*, and a lookahead that stops trapping is
not automatically a better one.

Both arms are a real party fight -- allies with real damage, every round,
in the ground truth. They differ only in what the policy is *shown*:

    blind   the allies are stripped from the state the policy plans
            against, which is exactly today's behaviour
    aware   the policy sees them, so its rollouts count their damage

Paired seeds, so the comparison rides on the difference rather than on
which arm drew the better shuffles.

    python party_pressure_probe.py [n]

RESULT (18 cells: 3 decks x {2,4} wizards x 3 boards, n=60 paired):

    kill rate   party of 2: +0.74 points mean, range -11.7 to +20.0
                party of 4: +0.00 -- every cell saturated at 100%
    turns       party of 2: +0.147 saved per fight
                party of 4: +0.294 saved per fight (~4% of a 7-turn fight),
                            but 3 of the 9 cells were NEGATIVE

So: inside the noise on kill rate, with two contested storm cells losing
6.7 and 11.7 points, and a turn-count gain that does not hold its sign.
**Not shipped.** `minion_power` stays 0.0 and the allies stay inert, and
this file is the reason -- so the zero reads as a measured decision
rather than as an oversight waiting to be tidied up.

What the party DOES coordinate is the first move of each round, where the
effect is large and measured: see `deimos_bridge/hivemind.py`.
"""
import copy
import functools
import sys

from data_full import load_spells_full
from w101_sim import Actor, Boss, Rules, Sim, evaluate_paired

from deimos_bridge.policies import greedy_ttk

print = functools.partial(print, flush=True)

CARDS = load_spells_full()

DECKS = {
    "fire": ["Fire Cat"] * 3 + ["Fireblade"] * 3 + ["Fire Elf"] * 2
            + ["Sunbird"] * 2,
    "ice": ["Frost Beetle"] * 3 + ["Iceblade"] * 2 + ["Ice Trap"] * 2
           + ["Snow Serpent"] * 3,
    "storm": ["Thunder Snake"] * 3 + ["Stormblade"] * 3
             + ["Lightning Bats"] * 3,
}

# Scaled for a party. A board one wizard finds hard is a board four
# wizards clear every time, and a cell at 100% ranks nothing -- the same
# saturation lesson `gui.app.TrainWorker.compare` records for the solo
# probes.
BOARDS = [
    [(1400, "death"), (1400, "death")],
    [(2400, "myth"), (900, "myth")],
    [(900, "ice"), (900, "ice"), (900, "ice")],
]

#: what one teammate deals per round, and what the whole board deals
#: before it is split between the wizards standing in front of it
ALLY_OUTPUT = 120.0
BOARD_OUTPUT = 160.0


def blind_to_party(policy):
    """The policy, shown a board with no teammates on it.

    A shallow copy: the player and the enemies are the same objects, so
    only the ally list differs. `_rollout` deep-copies whatever it is
    handed, so its rollouts inherit the empty list.
    """
    def strat(sim, s):
        without = copy.copy(s)
        without.allies = []
        return policy(sim, without)
    return strat


def party_sim(school, deck, board, incoming, n_allies, power,
              player_hp=1500):
    boss = Boss(name="Mob0", hp=board[0][0], school=board[0][1], dmg=incoming)
    extra = [Boss(name=f"Mob{i}", hp=hp, school=sc, dmg=incoming)
             for i, (hp, sc) in enumerate(board[1:], 1)]
    sim = Sim(CARDS, deck, school, boss, enemies=extra, rules=Rules(),
              player_hp=player_hp)
    base = sim.new_state

    def new_state():
        s = base()
        # Threat far below the player's, deliberately. `dmg` here is
        # already this wizard's SHARE of the board's output -- which is
        # what `_estimate_incoming` measures live -- and the sim's threat
        # rule is winner-takes-all rather than a spread, so letting the
        # allies enter the contest would discount the incoming twice.
        s.allies = [Actor(name=f"Ally{i}", school="balance", hp=player_hp,
                          max_hp=player_hp, team=0, minion_power=power,
                          threat=-1e9)
                    for i in range(n_allies)]
        return s

    sim.new_state = new_state
    return sim


def main(n=60):
    print(f"paired seeds n={n} per cell\n")
    print(f"{'deck':6} {'party':>5} {'board':>26} {'blind':>8} {'aware':>8} "
          f"{'delta':>7}  {'ttk blind':>9} {'ttk aware':>9}")
    kills, turns, cells = {}, {}, {}
    for school, deck in DECKS.items():
        for size in (2, 4):
            for board in BOARDS:
                sim = party_sim(school, deck, board,
                                max(25, int(BOARD_OUTPUT / size)),
                                size - 1, ALLY_OUTPUT)
                stats = evaluate_paired(
                    sim, {"blind": blind_to_party(greedy_ttk()),
                          "aware": greedy_ttk()}, n=n)
                b, a = stats["blind"], stats["aware"]
                delta = (a["win_rate"] - b["win_rate"]) * 100
                saved = b["mean_ttk"] - a["mean_ttk"]
                label = "+".join(str(hp) for hp, _s in board)
                print(f"{school:6} {size:>5} {label:>26} "
                      f"{b['win_rate'] * 100:7.1f}% {a['win_rate'] * 100:7.1f}%"
                      f" {delta:+6.1f}  {b['mean_ttk']:9.2f} "
                      f"{a['mean_ttk']:9.2f}")
                kills[size] = kills.get(size, 0.0) + delta
                turns[size] = turns.get(size, 0.0) + (saved if saved == saved
                                                      else 0.0)
                cells[size] = cells.get(size, 0) + 1
    print()
    for size in sorted(cells):
        n_cells = cells[size]
        print(f"party of {size}: kill rate {kills[size] / n_cells:+.2f} "
              f"points, turns saved {turns[size] / n_cells:+.3f} per fight, "
              f"over {n_cells} cells")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
