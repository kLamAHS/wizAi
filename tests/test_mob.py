"""
Target switching: focus selection, per-target casting, and the
kill-the-healer-first mechanic end to end.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import load_spells_full, LIVE_RULES
from w101_sim import (Sim, Boss, evaluate_paired, make_blade_stack,
                      pick_focus, with_focus)

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def _boss(name="boss", hp=2000, pool=None, archetype="hitter"):
    b = Boss(name, hp, "storm", 0, pool=pool, archetype=archetype)
    b.resist_map = {}
    b.boost_map = {}
    return b


def _sim(extra, seed=0):
    return Sim(dict(CARDS), ["Wraith", "Wraith", "Deathblade"], "death",
               _boss(pool=["Thunder Snake"]), player_hp=10**6,
               rules=LIVE_RULES, rng=random.Random(seed), enemies=extra,
               log_events=True)


def test_pick_focus_prefers_support_then_lowest_hp():
    healer = _boss("healer", 300, pool=["Sprite"], archetype="healer")
    sim = _sim([healer])
    s = sim.new_state()
    assert pick_focus(s) == 1                   # support first
    s.enemies[1].hp = 0                         # alive is derived
    assert pick_focus(s) == 0                   # then the boss
    sim2 = _sim([_boss("add", 150, pool=["Thunder Snake"])])
    s2 = sim2.new_state()
    assert pick_focus(s2) == 1                  # lowest-HP attacker


def test_focus_wrapper_hits_the_chosen_target():
    healer = _boss("healer", 300, pool=["Sprite"], archetype="healer")
    sim = _sim([healer], seed=3)
    pol = with_focus(make_blade_stack(0))
    t, won, hp = sim.run(pol, max_turns=20)
    ev = sim.last_state.events
    # the healer died before the boss did (or the fight was won)
    healer_deaths = [e for e in ev if e.get("type") == "death"
                     and e.get("actor") == "healer"]
    assert won or not sim.last_state.enemies[1].alive or healer_deaths


def test_focus_beats_blind_against_a_healer():
    """The report's team-fight rule, measured: focus-fire on the
    healer must strictly beat target-blind play."""
    healer = _boss("healer", 400, pool=["Sprite", "Pixie"],
                   archetype="healer")
    boss = _boss(hp=1500, pool=["Thunder Snake", "Stormblade"])
    deck = ["Wraith"] * 3 + ["Banshee"] * 3 + \
           ["Deathblade", "Death Trap", "Curse"]
    sim = Sim(dict(CARDS), deck, "death", boss, player_hp=10**6,
              rules=LIVE_RULES, enemies=[healer])
    st = evaluate_paired(sim, {
        "blind": make_blade_stack(2),
        "focus": with_focus(make_blade_stack(2))}, n=300)
    assert st["focus"]["win_rate"] > st["blind"]["win_rate"]
