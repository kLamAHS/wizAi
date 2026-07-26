"""
Property-based invariants (research-notes §"tests your simulator needs"):
monotonicity, conservation, reproducibility, bound soundness. These guard
the ML results: a learner exploiting a violation of any of these is
exploiting a simulator bug, not a strategy.
"""
import random
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w101_sim import (load_cards, Sim, Boss, Rules, Hanging,
                      strat_nuke_asap, make_blade_stack, evaluate_paired,
                      max_remaining_damage, provably_unwinnable,
                      CheatScript, CheatRule)
from bosses import BOSSES, DECKS

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_cards(str(ROOT / "cards_clean.json"))


class R0(random.Random):
    def random(self):
        return 0.0


def fresh(deck=("Fireblade",), school="fire", boss=None, **kw):
    boss = boss or Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), list(deck), school, boss, rng=R0(), **kw)
    return sim, sim.new_state()


def hit_damage(school, card_name, blade=None, pierce=0.0, boss_ward=None):
    sim, s = fresh(school=school)
    s.player.pierce = pierce
    s.player.pow_pips = 7
    if boss_ward is not None:
        s.enemies[0].wards.append(boss_ward)
    if blade:
        b = sim.cards[blade]
        s.hand.append(b)
        assert sim.can_cast(s, b)
        sim.cast(s, b)
        s.player.pow_pips = 7
    c = sim.cards[card_name]
    s.hand.append(c)
    hp0 = s.boss_hp
    sim.cast(s, c)
    return hp0 - s.boss_hp


# ---------------------------------------------------------------- monotonicity

@pytest.mark.parametrize("school,card,blade", [
    ("fire", "Fire Shark", "Fireblade"),
    ("fire", "Fire Elf", "Fireblade"),          # DoT initial portion
    ("myth", "Minotaur", "Mythblade"),          # multi-hit
    ("death", "Vampire", "Deathblade"),         # drain
])
def test_blade_never_reduces_damage(school, card, blade):
    assert hit_damage(school, card, blade=blade) >= \
        hit_damage(school, card) - 1e-9


def test_pierce_monotone_against_resist_and_shield():
    ward = lambda: Hanging("Tower Shield", "ward", "damage",
                           percent=-0.50, schools=None)
    prev = -1.0
    for pierce in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        d = hit_damage("fire", "Fire Shark", pierce=pierce,
                       boss_ward=ward())
        assert d >= prev - 1e-9
        prev = d


# ---------------------------------------------------------------- conservation

def test_deck_conservation_through_episode():
    b = BOSSES["Jade Oni"]
    sim = Sim(dict(CARDS), DECKS["fire"]["oneshot"], "fire",
              Boss(b.name, b.hp, b.school, 0), player_hp=10**9,
              rng=random.Random(5))
    total0 = len(sim.decklist)

    inner = make_blade_stack(3)
    def counting(sim_, st):
        assert len(st.hand) + len(st.deck) + len(st.player.graveyard) \
            == total0
        return inner(sim_, st)
    sim.run(counting, max_turns=30)


def test_tc_conservation():
    boss = Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), ["Fireblade"] * 3, "fire", boss, rng=R0(),
              sideboard=["Fairy@tc"] * 4)
    s = sim.new_state()
    sim.draw_tc(s, 2)
    held = len(s.hand) + len(s.deck) + len(s.player.graveyard) + \
        len(s.player.sideboard)
    assert held == 3 + 4


def test_fuel_charges_only_consumed_by_matching_hits():
    sim, s = fresh(school="fire")
    s.player.pow_pips = 7
    fuel = sim.cards["Fuel"]
    s.hand.append(fuel)
    sim.cast(s, fuel)
    # off-school (myth) hit must not touch the fire-only Fuel charges
    myth_hit = sim.cards["Cyclops"]
    s.hand.append(myth_hit)
    s.player.pow_pips = 7
    sim.cast(s, myth_hit)
    assert s.traps["Fuel"][1] == 3
    fire_hit = sim.cards["Fire Cat"]
    s.hand.append(fire_hit)
    s.player.pow_pips = 7
    sim.cast(s, fire_hit)
    assert s.traps["Fuel"][1] == 2


# ---------------------------------------------------------------- reproducibility

def test_same_seed_same_event_log():
    b = BOSSES["Krokopatra"]
    logs = []
    for _ in range(2):
        sim = Sim(dict(CARDS), DECKS["fire"]["oneshot"], "fire",
                  Boss(b.name, b.hp, b.school, 0), player_hp=10**9,
                  rng=random.Random(99), log_events=True)
        s = sim.new_state()
        pol = make_blade_stack(3)
        while s.turn < 20 and s.boss_hp > 0:
            c = pol(sim, s)
            if c is not None and sim.can_cast(s, c):
                sim.cast(s, c)
            if s.boss_hp <= 0:
                break
            sim.end_round(s)
        logs.append(s.events)
    assert logs[0] == logs[1]
    assert any(e["type"] == "damage_applied" for e in logs[0])


def test_evaluate_paired_is_deterministic():
    b = BOSSES["Rattlebones"]
    sim = Sim(dict(CARDS), DECKS["fire"]["speed"], "fire",
              Boss(b.name, b.hp, b.school, 0), player_hp=10**9)
    pols = {"nuke": strat_nuke_asap, "stack": make_blade_stack(2)}
    a = evaluate_paired(sim, pols, n=100)
    b2 = evaluate_paired(sim, pols, n=100)
    assert a == b2
    assert a["nuke"]["ruleset"] == "w101-pve-classic-0.3"
    assert a["nuke"]["p90_ttk"] >= a["nuke"]["median_ttk"]


# ---------------------------------------------------------------- damage bound

def test_no_impossible_kill_classifier():
    b = BOSSES["Jade Oni"]
    boss = Boss(b.name, b.hp, b.school, 0)
    # the fire speed deck cannot cover 4000 HP once fizzled cards are lost
    sim = Sim(dict(CARDS), DECKS["fire"]["speed"], "fire", boss,
              player_hp=10**9, rng=random.Random(0))
    s = sim.new_state()
    assert provably_unwinnable(sim, s)
    # the oneshot deck can
    sim2 = Sim(dict(CARDS), DECKS["fire"]["oneshot"], "fire", boss,
               player_hp=10**9, rng=random.Random(0))
    s2 = sim2.new_state()
    assert not provably_unwinnable(sim2, s2)


def test_bound_dominates_realized_damage():
    """The optimistic bound must never be exceeded by actual play."""
    b = BOSSES["Krokopatra"]
    for seed in range(5):
        sim = Sim(dict(CARDS), DECKS["fire"]["oneshot"], "fire",
                  Boss(b.name, b.hp, b.school, 0), player_hp=10**9,
                  rng=random.Random(seed))
        s = sim.new_state()
        bound = max_remaining_damage(sim, s)
        pol = make_blade_stack(4)
        hp0 = s.boss_hp
        while s.turn < 30 and s.boss_hp > 0:
            c = pol(sim, s)
            if c is not None and sim.can_cast(s, c):
                sim.cast(s, c)
            if s.boss_hp <= 0:
                break
            sim.end_round(s)
        assert hp0 - s.boss_hp <= bound + 1e-6


# ---------------------------------------------------------------- rules objects

def test_crit_resolver_is_pluggable():
    rules = Rules(crit_resolver=lambda a, d, rng, r: 3.0)
    boss = Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), ["Fireblade"], "fire", boss, rng=R0(),
              rules=rules)
    s = sim.new_state()
    s.player.pow_pips = 7
    c = sim.cards["Fire Shark"]
    s.hand.append(c)
    hp0 = s.boss_hp
    sim.cast(s, c)
    assert s.boss_hp == pytest.approx(hp0 - 405 * 3.0)


def test_flat_damage_and_flat_resist():
    boss = Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), ["Fireblade"], "fire", boss, rng=R0())
    s = sim.new_state()
    s.player.flat_damage = 50
    s.enemies[0].flat_resist = 30
    s.player.pow_pips = 7
    c = sim.cards["Fire Shark"]
    s.hand.append(c)
    hp0 = s.boss_hp
    sim.cast(s, c)
    assert s.boss_hp == pytest.approx(hp0 - (405 + 50 - 30))


def test_cheat_cooldown_limits_firing():
    script = CheatScript([CheatRule(
        event="after_player_cast", when={"card_kind": "blade"},
        ops=[{"op": "hit", "amount": 100, "tgt": "enemy"}], cooldown=3)])
    boss = Boss("CD", 10000, "life", 0, cheat=script)
    sim, s = fresh(boss=boss)
    s.player.pow_pips = 7
    for i, name in enumerate(["Fireblade", "Fireblade@tc", "Mythblade"]):
        from w101_sim import expand_variants
        expand_variants(sim.cards, [name])
        c = sim.cards[name]
        s.hand.append(c)
        s.player.pow_pips = 7
        sim.cast(s, c)
    # three blade casts in the same round: cooldown allows only the first
    assert s.player.hp == pytest.approx(3000 - 100)


# ---------------------------------------------------------------- search

def test_search_policy_runs_and_beats_nothing_silly():
    from search_policy import make_search_policy
    b = BOSSES["Rattlebones"]
    sim = Sim(dict(CARDS), DECKS["fire"]["speed"], "fire",
              Boss(b.name, b.hp, b.school, 0), player_hp=10**9,
              rng=random.Random(3))
    t, won, _ = sim.run(make_search_policy(k=3), max_turns=15)
    assert won


def test_search_policy_does_not_mutate_real_state():
    from search_policy import make_search_policy
    b = BOSSES["Krokopatra"]
    sim = Sim(dict(CARDS), DECKS["fire"]["oneshot"], "fire",
              Boss(b.name, b.hp, b.school, 0), player_hp=10**9,
              rng=random.Random(3))
    s = sim.new_state()
    hand0 = [c.name for c in s.hand]
    hp0, pips0 = s.boss_hp, s.player.pip_slots
    make_search_policy(k=2)(sim, s)      # decide only, no execution
    assert [c.name for c in s.hand] == hand0
    assert s.boss_hp == hp0 and s.player.pip_slots == pips0
