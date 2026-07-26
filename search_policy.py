"""
Belief-state search baseline: determinized rollout search.

The deck's COMPOSITION is known to the player (deck tracking is perfect
information); only the DRAW ORDER is hidden. That makes the fight a POMDP
whose belief state is uniform over permutations of the remaining pile — so
the textbook baseline is determinization: for each candidate action, sample
K shuffles of the hidden remainder, play the action, roll the fight out
with a base heuristic, and keep the action with the best mean outcome.

No training, interpretable, and a strong reference point above the tabular
learner: any gap between this and RL is planning ability, not knowledge.
Score convention matches the RL reward: -turns on a win, -(turns + fail
penalty) on a defeat/timeout.
"""
import copy
import random

from w101_sim import State, make_blade_stack


def _clone_actor(a):
    b = copy.copy(a)
    b.charms = [copy.copy(h) for h in a.charms]
    b.wards = [copy.copy(h) for h in a.wards]
    b.over_time = [copy.copy(e) for e in a.over_time]
    b.damage_bonus = dict(a.damage_bonus)
    b.resist = dict(a.resist)
    b.boost = dict(a.boost)
    b.hand = list(a.hand)
    b.deck = list(a.deck)
    b.sideboard = list(a.sideboard)
    b.graveyard = list(a.graveyard)
    b.fresh_tc = list(a.fresh_tc)
    if a.cheat:
        b.cheat = type(a.cheat)([copy.copy(r) for r in a.cheat.rules])
    return b


def clone_state(s):
    ns = State(_clone_actor(s.player), [_clone_actor(e) for e in s.enemies],
               [_clone_actor(m) for m in s.allies])
    ns.bubble = copy.copy(s.bubble) if s.bubble else None
    ns.turn = s.turn
    return ns


def _rollout(sim, s, pol, max_turns, fail):
    """Finish the duel from mid-turn (candidate already applied)."""
    while True:
        if not any(e.alive for e in s.enemies):
            return -(s.turn + 1)
        sim.end_round(s)
        if s.player.hp <= 0:
            return -(s.turn + fail)
        if not any(e.alive for e in s.enemies):
            return -s.turn
        if s.turn >= max_turns:
            return -(max_turns + fail)
        sim._tick_over_time(s, s.player)
        if s.player.hp <= 0:
            return -(s.turn + fail)
        if s.player.stunned > 0:
            s.player.stunned -= 1
        else:
            c = pol(sim, s)
            if c is not None and sim.can_cast(s, c):
                sim.cast(s, c)


def make_search_policy(base=None, k=6, fail_penalty=25.0, max_turns=40):
    """Determinized one-ply search: argmax over castable cards (+ pass) of
    the mean rollout return across k sampled draw orders."""
    base_pol = base or make_blade_stack(3)

    def policy(sim, s):
        cands, seen = [None], set()
        for c in s.hand:
            if c.name not in seen and sim.can_cast(s, c):
                seen.add(c.name)
                cands.append(c)
        if len(cands) == 1:
            return None
        best, best_v = None, -float("inf")
        for cand in cands:
            v = 0.0
            for _ in range(k):
                ds = clone_state(s)
                dsim = copy.copy(sim)
                dsim.rng = random.Random(sim.rng.getrandbits(32))
                dsim.log_events = False
                dsim.rng.shuffle(ds.player.deck)   # determinize hidden pile
                if cand is not None:
                    m = next(c for c in ds.player.hand
                             if c.name == cand.name)
                    if dsim.can_cast(ds, m):
                        dsim.cast(ds, m)
                v += _rollout(dsim, ds, base_pol, max_turns, fail_penalty)
            v /= k
            if v > best_v:
                best_v, best = v, cand
        return best
    return policy


if __name__ == "__main__":
    from w101_sim import load_cards, Boss, Sim, evaluate_paired
    from bosses import BOSSES, DECKS

    cards = load_cards("cards_clean.json")
    for school, bname, dname in [("fire", "Jade Oni", "oneshot"),
                                 ("ice", "Prince Gobblestone", "prism")]:
        b = BOSSES[bname]
        sim = Sim(cards, DECKS[school][dname], school,
                  Boss(b.name, b.hp, b.school, 0), player_hp=10**9)
        stats = evaluate_paired(
            sim, {"blade-stack(3)": make_blade_stack(3),
                  "search(k=6)": make_search_policy(k=6)}, n=400)
        print(f"\n{school} vs {bname} [{dname}] (paired seeds, n=400)")
        for name, st in stats.items():
            print(f"  {name:<16} win {st['win_rate']*100:5.1f}%  "
                  f"mean {st['mean_ttk']:5.2f}  med {st['median_ttk']:3.0f}  "
                  f"p90 {st['p90_ttk']:3.0f}")
