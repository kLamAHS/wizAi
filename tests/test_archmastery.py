"""
School pips / archmastery: generation, the own-school lock, spend
order, and zero-impact when the era is off.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace

from data_full import load_spells_full, LIVE_RULES
from rulesets import AM_MASTERY, ARCHMASTERY, stats_for
from w101_sim import Sim, Boss, evaluate_paired, make_blade_stack

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))
ALWAYS = replace(LIVE_RULES, archmastery=1.0)


def _boss(hp=2000):
    b = Boss("dummy", hp, "death", 0)
    b.resist_map = {}
    b.boost_map = {}
    return b


def _sim(rules, school="fire", deck=None):
    return Sim(dict(CARDS), deck or ["Sunbird", "Kraken"], school,
               _boss(), player_hp=10**9, rules=rules,
               rng=random.Random(0))


def test_archmastery_generates_school_pips():
    sim = _sim(ALWAYS)
    s = sim.new_state()
    for _ in range(4):
        sim.gain_pip(s)
    assert s.school_pips >= 4               # 1 from new_state + 4
    assert s.norm_pips == 0 and s.pow_pips == 0


def test_school_pips_pay_two_for_own_school_only():
    sim = _sim(ALWAYS)                      # fire wizard
    s = sim.new_state()
    s.norm_pips = s.pow_pips = 0
    s.school_pips = 2                       # = 4 pips of fire
    fire = next(c for c in s.hand if c.name == "Sunbird")     # 3 pips
    storm = next(c for c in s.hand if c.name == "Kraken")     # 4 pips
    assert sim.afford(s, fire)
    assert not sim.afford(s, storm)         # locked out of the splash


def test_mastery_does_not_unlock_school_pips():
    """The amulet grants power-pip VALUE, not school-pip access — a
    storm mastery still cannot spend fire school pips on Kraken."""
    sim = _sim(replace(AM_MASTERY, archmastery=1.0))
    s = sim.new_state()
    s.norm_pips = s.pow_pips = 0
    s.school_pips = 3
    storm = next(c for c in s.hand if c.name == "Kraken")
    assert not sim.afford(s, storm)


def test_school_pips_are_spent_first():
    """They are worthless on anything else, so an own-school cast
    should burn them before generic pips."""
    sim = _sim(ALWAYS)
    s = sim.new_state()
    s.norm_pips, s.pow_pips, s.school_pips = 2, 2, 2
    fire = next(c for c in s.hand if c.name == "Sunbird")     # 3 pips
    sim.spend(s, fire)
    assert s.school_pips == 1               # 1 school pip (2) + 1 norm
    assert s.norm_pips == 1 and s.pow_pips == 2


def test_school_pips_count_in_the_slot_cap():
    sim = _sim(ALWAYS)
    s = sim.new_state()
    s.norm_pips, s.pow_pips, s.school_pips = 2, 2, 3
    assert s.pip_slots == 7
    before = (s.norm_pips, s.pow_pips, s.school_pips)
    sim.gain_pip(s)                         # rack is full
    assert (s.norm_pips, s.pow_pips, s.school_pips) == before


def test_xpip_consumes_school_pips_for_own_school():
    sim = _sim(ALWAYS, school="storm", deck=["Tempest"])
    s = sim.new_state()
    s.norm_pips, s.pow_pips, s.school_pips = 1, 1, 2
    x = next(c for c in s.hand if c.name == "Tempest")
    eff = sim.spend(s, x)
    assert eff == 1 + 2 + 4                 # norm + power(2) + school(4)
    assert s.school_pips == 0


def test_archmastery_off_is_bit_identical():
    deck = ["Sunbird", "Sunbird", "Sunbird", "Fireblade", "Fire Trap"]
    a = Sim(dict(CARDS), deck, "fire", _boss(500), player_hp=10**9,
            rules=LIVE_RULES)
    b = Sim(dict(CARDS), deck, "fire", _boss(500), player_hp=10**9,
            rules=replace(LIVE_RULES, archmastery=0.0))
    r1 = evaluate_paired(a, {"p": make_blade_stack(2)}, n=250)["p"]
    r2 = evaluate_paired(b, {"p": make_blade_stack(2)}, n=250)["p"]
    assert r1["win_rate"] > 0.5            # a real fight, not 0-vs-0
    for k in ("win_rate", "mean_ttk", "p90_ttk"):
        assert r1[k] == r2[k] or (r1[k] != r1[k] and r2[k] != r2[k])
    assert ARCHMASTERY.archmastery > 0 and LIVE_RULES.archmastery == 0


def test_school_gear_is_own_school_only():
    st = stats_for(ARCHMASTERY, school="fire", school_gear=True)
    assert st["damage"] == {"fire": 0.50}
    assert stats_for(ARCHMASTERY, school="fire") == {}
