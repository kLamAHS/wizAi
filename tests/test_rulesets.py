"""
Post-classic rulesets: mastery pips, rating criticals, and the
guarantee that classic-era behavior is untouched.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace

from data_full import load_spells_full, LIVE_RULES
from rulesets import CLASSIC, CRIT_ERA, MASTERY, MODERN, ERAS, stats_for
from w101_sim import (Sim, Boss, evaluate_paired, make_blade_stack,
                      make_rating_crit)

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def _boss(hp=2000):
    b = Boss("dummy", hp, "death", 0)
    b.resist_map = {}
    b.boost_map = {}
    return b


def test_every_era_has_a_distinct_id():
    ids = [r.ruleset_id for r in ERAS.values()]
    assert len(ids) == len(set(ids))


def test_mastery_gives_full_power_pips_off_school():
    """A storm nuke costs a fire wizard double pips — unless the
    ruleset grants a storm mastery amulet."""
    deck = ["Kraken"]                      # storm, 4 pips
    plain = Sim(dict(CARDS), deck, "fire", _boss(), player_hp=10**9,
                rules=LIVE_RULES)
    mast = Sim(dict(CARDS), deck, "fire", _boss(), player_hp=10**9,
               rules=MASTERY)
    for sim, expect in ((plain, False), (mast, True)):
        s = sim.new_state()
        s.norm_pips, s.pow_pips = 0, 2     # 2 power pips
        card = next(c for c in s.hand if c.name == "Kraken")
        assert sim.afford(s, card) is expect


def test_mastery_spends_power_pips_two_for_one():
    sim = Sim(dict(CARDS), ["Kraken"], "fire", _boss(), player_hp=10**9,
              rules=MASTERY)
    s = sim.new_state()
    s.norm_pips, s.pow_pips = 0, 3
    card = next(c for c in s.hand if c.name == "Kraken")
    sim.spend(s, card)
    assert (s.norm_pips, s.pow_pips) == (0, 1)   # 4 pips = 2 power


def test_mastery_does_not_leak_to_other_schools():
    sim = Sim(dict(CARDS), ["Sunbird"], "death", _boss(),
              player_hp=10**9, rules=MASTERY)    # mastery is storm
    s = sim.new_state()
    s.norm_pips, s.pow_pips = 0, 2
    card = next(c for c in s.hand if c.name == "Sunbird")
    assert not sim.afford(s, card)               # fire still costs full


def test_rating_crit_is_monotone_and_bounded():
    res = make_rating_crit()
    rules = replace(CLASSIC, crit_multiplier=2.0)

    class A:
        crit_chance = 0.0
        block_chance = 0.0

    a, d = A(), A()
    rng = random.Random(0)
    assert res(a, d, rng, rules) == 1.0          # no rating, no crit
    a.crit_chance = 400.0
    hits = [res(a, d, random.Random(i), rules) for i in range(400)]
    rate_hi = sum(h > 1.0 for h in hits) / len(hits)
    a.crit_chance = 100.0
    hits = [res(a, d, random.Random(i), rules) for i in range(400)]
    rate_lo = sum(h > 1.0 for h in hits) / len(hits)
    assert rate_hi > rate_lo                     # more rating, more crits
    assert all(1.0 <= h <= 2.0 for h in hits)    # never exceeds the cap


def test_block_rating_reduces_crit_damage():
    res = make_rating_crit()
    rules = replace(CLASSIC, crit_multiplier=2.0)

    class A:
        crit_chance = 10**6                      # always crits
        block_chance = 0.0

    a = A()
    unblocked = A()
    unblocked.crit_chance = 0.0
    mults = []
    for br in (0.0, 300.0):
        d = A()
        d.crit_chance = 0.0
        d.block_chance = br
        got = [res(a, d, random.Random(i), rules) for i in range(300)]
        crits = [m for m in got if m > 1.0]
        mults.append(sum(crits) / max(len(crits), 1))
    assert mults[1] < mults[0]                   # block shaves damage


def test_classic_behavior_is_bit_identical():
    """The new knobs must not perturb any existing classic result."""
    deck = ["Wraith", "Wraith", "Deathblade", "Feint"]
    a = Sim(dict(CARDS), deck, "death", _boss(1200), player_hp=10**9,
            rules=CLASSIC)
    b = Sim(dict(CARDS), deck, "death", _boss(1200), player_hp=10**9,
            rules=replace(CLASSIC))
    r1 = evaluate_paired(a, {"p": make_blade_stack(2)}, n=250)["p"]
    r2 = evaluate_paired(b, {"p": make_blade_stack(2)}, n=250)["p"]
    assert r1 == r2
    assert CLASSIC.mastery_school is None and CLASSIC.crit_resolver is None


def test_stats_for_only_arms_crit_eras():
    assert stats_for(CLASSIC) == {}
    assert stats_for(MASTERY) == {}              # mastery has no crits
    assert stats_for(CRIT_ERA)["crit"] > 0
    assert stats_for(MODERN)["crit"] > 0
