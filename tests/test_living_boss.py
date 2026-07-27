"""
Player stat curves (report anchors) and the living-boss caster layer
(pool + archetype AI + modeled imperfection).
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import load_spells_full, LIVE_RULES
from player_curves import HP_ANCHORS, school_hp, base_power_pip
from w101_sim import Sim, Boss, evaluate, make_blade_stack

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def test_hp_curve_hits_report_anchors_exactly():
    for school, (lo, hi) in HP_ANCHORS.items():
        assert school_hp(school, 1) == lo
        assert school_hp(school, 120) == hi
        assert school_hp(school, 180) == hi     # clamped, not invented
    assert school_hp("ice", 60) > school_hp("storm", 60)
    hp = [school_hp("fire", l) for l in range(1, 121)]
    assert hp == sorted(hp)                     # monotone between anchors


def test_power_pip_curve_breakpoints():
    assert base_power_pip(1) == 0.0
    assert base_power_pip(9) == 0.0
    assert base_power_pip(10) == 0.0            # "begins" at 10: first
    assert base_power_pip(11) == pytest.approx(0.01)   # point at 11
    assert base_power_pip(20) == pytest.approx(0.10)
    assert base_power_pip(30) == pytest.approx(0.20)
    assert base_power_pip(50) == 0.40
    assert base_power_pip(180) == 0.40          # base cap, gear excluded


def _living(pool, archetype="hitter", hp=2000, school="storm",
            discipline=1.0, player_hp=10**9, seed=0):
    b = Boss("living", hp, school, 0, pool=pool, archetype=archetype,
             discipline=discipline)
    b.resist_map = {}
    b.boost_map = {}
    return Sim(dict(CARDS), ["Wraith", "Wraith", "Deathblade"], "death",
               b, player_hp=player_hp, rules=LIVE_RULES,
               rng=random.Random(seed), log_events=True)


def _events(sim, rounds=10):
    s = sim.new_state()
    for _ in range(rounds):
        sim.end_round(s)
        if s.player.hp <= 0:
            break
    return s, [e for e in s.events if e["type"].startswith("enemy")]


def test_unknown_pool_spell_fails_loudly():
    with pytest.raises(ValueError):
        _living(["Totally Fake Spell"])


def test_hitter_blades_then_nukes():
    """Discipline 1.0 hitter: saves for the top nuke behind a blade —
    the report's hitter archetype — and actually hurts the player."""
    sim = _living(["Kraken", "Stormblade", "Storm Shark"],
                  player_hp=10**6, seed=3)
    s, ev = _events(sim, rounds=12)
    casts = [e["card"] for e in ev if e["type"] == "enemy_cast"]
    assert "Stormblade" in casts                # setup happened
    assert any(c in ("Kraken", "Storm Shark") for c in casts)
    assert casts.index("Stormblade") < max(
        i for i, c in enumerate(casts) if c != "Stormblade")
    assert any(e["type"] == "enemy_pass" for e in ev)   # saved pips
    assert s.player.hp < 10**6                  # damage landed


def test_no_duplicate_blade_stacking():
    """Blade hangs once; while it hangs, the second copy is illegal.
    Pool includes an unaffordable top nuke so the save branch actually
    reaches for the blade."""
    sim = _living(["Stormblade", "Storm Owl"], player_hp=10**6, seed=1)
    s = sim.new_state()
    for _ in range(6):
        sim.end_round(s)
        assert len([h for h in s.enemies[0].charms
                    if h.name == "Stormblade"]) <= 1
    casts = [e["card"] for e in s.events if e["type"] == "enemy_cast"]
    assert "Stormblade" in casts


def test_full_rack_upgrades_out_of_livelock():
    """Regression: an own-school 10-pip nuke needs 3+ power pips in a
    7-slot rack. Without the full-rack white->power upgrade the boss
    could freeze into a forever-pass; with it, Storm Owl always fires
    eventually."""
    for seed in range(10):
        sim = _living(["Storm Owl"], seed=seed, player_hp=10**6)
        s = sim.new_state()
        casts = []
        for _ in range(40):
            sim.end_round(s)
            casts = [e["card"] for e in s.events
                     if e["type"] in ("enemy_cast", "enemy_fizzle")]
            if "Storm Owl" in casts:
                break
        assert "Storm Owl" in casts, f"livelock at seed {seed}"


def test_healer_archetype_heals_the_injured():
    sim = _living(["Sprite"], archetype="healer", hp=3000, school="life",
                  seed=2, player_hp=10**6)
    s = sim.new_state()
    s.enemies[0].hp = 900                       # badly hurt
    for _ in range(6):
        sim.end_round(s)
    assert s.enemies[0].hp > 900                # it healed itself


def test_healer_heals_the_hurt_teammate_not_itself():
    """The role-minion pattern: a full-HP acolyte must route its heal
    to the injured boss (enemy-side ally targeting)."""
    boss = Boss("hurt-boss", 2000, "storm", 0)
    boss.resist_map = {}
    boss.boost_map = {}
    acolyte = Boss("acolyte", 800, "life", 0, pool=["Sprite"],
                   archetype="healer", discipline=1.0)
    acolyte.resist_map = {}
    acolyte.boost_map = {}
    sim = Sim(dict(CARDS), ["Wraith"], "death", boss, player_hp=10**6,
              rules=LIVE_RULES, rng=random.Random(7), enemies=[acolyte],
              log_events=True)
    s = sim.new_state()
    s.enemies[0].hp = 400                       # boss badly hurt
    for _ in range(8):
        sim.end_round(s)
    assert s.enemies[0].hp > 400                # teammate got the heal
    assert s.enemies[1].hp == 800               # healer didn't waste it


def test_enemy_pip_spend_math():
    """Mutation guard: casting really deducts pips, power pips double
    for own school, and X-pip spells dump the whole rack."""
    sim = _living(["Kraken", "Tempest"], player_hp=10**6, seed=5)
    s = sim.new_state()
    e = s.enemies[0]
    kraken, tempest = CARDS["Kraken"], CARDS["Tempest"]
    e.norm_pips, e.pow_pips = 0, 2              # 4 effective, own school
    assert sim._enemy_afford(e, kraken)         # 4-pip spell
    sim._enemy_spend(e, kraken)
    assert (e.norm_pips, e.pow_pips) == (0, 0)
    e.norm_pips, e.pow_pips = 3, 2
    assert sim._enemy_cost(e, tempest) == 7     # X-pip = current rack
    assert sim._enemy_spend(e, tempest) == 7
    assert (e.norm_pips, e.pow_pips) == (0, 0)


def test_xpip_never_priced_as_free():
    """A tank poking with min-cost must prefer the 1-pip snake over
    dumping its rack into Tempest."""
    sim = _living(["Tempest", "Thunder Snake"], archetype="tank",
                  player_hp=10**6, seed=8)
    s = sim.new_state()
    e = s.enemies[0]
    e.norm_pips, e.pow_pips = 3, 0
    card, _ = sim._enemy_choose(s, e)
    assert card is not None and card.name == "Thunder Snake"


def test_discipline_zero_breaks_priority():
    """The report's imperfection knob: at discipline 0 the boss picks
    uniformly among legal casts, so the small poke DOES get cast; at
    discipline 1 the saver never wastes pips on it."""
    def casts_at(disc, seed):
        sim = _living(["Thunder Snake", "Kraken"], discipline=disc,
                      player_hp=10**6, seed=seed)
        s, ev = _events(sim, rounds=10)
        return [e["card"] for e in ev if e["type"] == "enemy_cast"]
    loose = sum("Thunder Snake" in casts_at(0.0, s) for s in range(8))
    strict = sum("Thunder Snake" in casts_at(1.0, s) for s in range(8))
    assert loose > 0
    assert strict == 0


def test_player_mantle_bites_the_living_boss():
    """A -100% accuracy charm hung on the boss must force a fizzle and
    be consumed — player debuff counterplay works against the caster."""
    from w101_sim import Hanging
    sim = _living(["Kraken"], player_hp=10**6, seed=9)
    s = sim.new_state()
    e = s.enemies[0]
    e.norm_pips = 6
    e.charms.append(Hanging("Test Mantle", "charm", "accuracy",
                            percent=-1.0, schools=None))
    sim._enemy_cast(s, e, CARDS["Kraken"])
    assert any(ev["type"] == "enemy_fizzle" for ev in s.events)
    assert not any(h.name == "Test Mantle" for h in e.charms)


def test_pass_is_a_real_action():
    """An expensive pool with no pips yet: the boss must pass, not
    crash, and the pass is logged (the report: passing is valid)."""
    sim = _living(["Storm Owl"], seed=4, player_hp=10**6)
    s = sim.new_state()
    sim.end_round(s)
    assert any(e["type"] == "enemy_pass" for e in s.events)


def test_legacy_flat_boss_unchanged():
    """No pool -> the old flat-damage path, byte-identical results."""
    b1 = Boss("flat", 800, "death", 120)
    b2 = Boss("flat", 800, "death", 120)
    s1 = Sim(dict(CARDS), ["Wraith"] * 3, "death", b1, player_hp=2000,
             rules=LIVE_RULES)
    s2 = Sim(dict(CARDS), ["Wraith"] * 3, "death", b2, player_hp=2000,
             rules=LIVE_RULES)
    from w101_sim import evaluate_paired
    r1 = evaluate_paired(s1, {"p": make_blade_stack(0)}, n=200)["p"]
    r2 = evaluate_paired(s2, {"p": make_blade_stack(0)}, n=200)["p"]
    assert r1 == r2


def test_living_boss_fizzles_at_low_accuracy():
    """Storm pool (70% acc) must fizzle sometimes — logged, harmless."""
    sim = _living(["Kraken"], seed=6, player_hp=10**6)
    fizz = 0
    for seed in range(20):
        sim.rng = random.Random(seed)
        s, ev = _events(sim, rounds=6)
        fizz += sum(1 for e in ev if e["type"] == "enemy_fizzle")
    assert fizz > 0
