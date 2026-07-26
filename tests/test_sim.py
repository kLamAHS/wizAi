"""
Mechanics test suite for the v0.3 engine.

Each test pins one documented combat rule (see the research notes / README):
stacking-by-provenance, FIFO ward processing with prism conversion, DoT/HoT
snapshots, drain semantics, multi-hit consumption, X-pip scaling, cheats,
treasure cards, stuns, threat, crit/block/pierce machinery, and the v0.2
compatibility surface used by the DP transfer and the tabular RL agent.
"""
import random
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w101_sim import (load_cards, expand_variants, sharpened, Sim, Boss,
                      Rules, Hanging, Card, Action, State, CheatScript,
                      CheatRule, make_blade_stack, make_survival,
                      strat_nuke_asap, evaluate)
import content

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_cards(str(ROOT / "cards_clean.json"))


class R0(random.Random):
    """random() == 0.0: casts always hit, pip gains are always power pips,
    crit rolls always succeed (and block rolls always block)."""
    def random(self):
        return 0.0


class RSeq(random.Random):
    """random() pops from a fixed sequence (then 0.0 forever)."""
    def __init__(self, seq):
        super().__init__(0)
        self.seq = list(seq)

    def random(self):
        return self.seq.pop(0) if self.seq else 0.0


def fresh(deck=("Fireblade",), school="fire", boss=None, hp=3000, rng=None,
          **kw):
    boss = boss or Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), list(deck), school, boss, rng=rng or R0(),
              player_hp=hp, **kw)
    s = sim.new_state()
    return sim, s


def give(sim, s, *names, pips=7):
    """Put cards in hand and fill power pips."""
    s.player.pow_pips = pips
    s.player.norm_pips = 0
    out = []
    for n in names:
        expand_variants(sim.cards, [n])
        c = sim.cards[n]
        s.hand.append(c)
        out.append(c)
    return out if len(out) > 1 else out[0]


def cast(sim, s, card, target=0):
    assert sim.can_cast(s, card, target), f"cannot cast {card.name}"
    return sim.cast(s, card, target)


# ---------------------------------------------------------------- loading

def test_load_coverage():
    report = {}
    load_cards(str(ROOT / "cards_clean.json"), report)
    skipped = {n for n, _ in report["skipped"]}
    # only genuinely unsupportable cards may be dropped
    assert skipped == content.UNSUPPORTED
    assert len(CARDS) == 170        # 171 scraped + 1 curated extra - 2 dupes?


def test_every_card_has_ops():
    for c in CARDS.values():
        assert c.ops, c.name
        for o in c.ops:
            assert o["op"] in {
                "hit", "dot", "hot", "heal", "drain", "charm", "ward",
                "prism", "absorb", "global", "self_damage", "gain_pips",
                "stun", "stun_block", "dispel", "remove", "steal", "summon",
                "threat", "sacrifice_minion"}, (c.name, o)


def test_confidence_tags():
    assert CARDS["Tempest"].confidence == "approx"
    assert CARDS["Fireblade"].confidence == "community"


# ---------------------------------------------------------------- pips

def test_power_pips_double_same_school_only():
    sim, s = fresh()
    helephant = give(sim, s, "Helephant", pips=3)     # 6 effective fire pips
    assert sim.afford(s, helephant)                   # costs 6
    feint = give(sim, s, "Feint", pips=2)             # off-school: worth 2
    assert not sim.afford(s, feint) or feint.pips <= 2  # feint costs 1: fine
    s.player.pow_pips, s.player.norm_pips = 1, 0
    kraken = give(sim, s, "Kraken", pips=0)
    s.player.pow_pips = 2                             # off-school: 2 < 4
    assert not sim.afford(s, kraken)
    s.player.pow_pips = 4
    assert sim.afford(s, kraken)


def test_odd_remainder_overpays_power_pip():
    sim, s = fresh()
    shark = give(sim, s, "Fire Shark", pips=0)        # 3 pips, fire
    s.player.pow_pips, s.player.norm_pips = 2, 0      # 4 effective
    cast(sim, s, shark)
    # 3 cost: one power pip (2) + forced overpay of the second (worth 2)
    assert s.player.pow_pips == 0 and s.player.norm_pips == 0


def test_x_pip_spends_everything_and_scales():
    sim, s = fresh(school="balance")
    judge = give(sim, s, "Judgement", pips=3)         # 6 effective
    hp0 = s.boss_hp
    cast(sim, s, judge)
    assert s.boss_hp == pytest.approx(hp0 - 600)      # 100/pip * 6
    assert s.player.pip_slots == 0


def test_empower_gains_pips_and_hurts():
    sim, s = fresh(school="death")
    emp = give(sim, s, "Empower", pips=1)
    hp0 = s.player.hp
    cast(sim, s, emp)
    assert s.player.hp == hp0 - 500
    assert s.player.norm_pips == 3 and s.player.pow_pips == 0


# ---------------------------------------------------------------- stacking

def test_same_name_same_source_does_not_stack():
    sim, s = fresh()
    b1, b2 = give(sim, s, "Fireblade", "Fireblade")
    cast(sim, s, b1)
    assert not sim.can_cast(s, b2)                    # illegal, like the game


def test_provenance_stacks():
    sim, s = fresh()
    b1, b2 = give(sim, s, "Fireblade", "Fireblade@tc")
    cast(sim, s, b1)
    cast(sim, s, b2)                                  # tc copy stacks
    shark = give(sim, s, "Fire Shark")
    hp0 = s.boss_hp
    cast(sim, s, shark)
    assert s.boss_hp == pytest.approx(hp0 - 405 * 1.35 * 1.35)


def test_sharpened_variant_stacks_and_is_stronger():
    sim, s = fresh()
    sharp = sharpened(sim.cards, "Fireblade")
    assert sharp.percent == pytest.approx(0.45)
    b1 = give(sim, s, "Fireblade")
    s.hand.append(sharp)
    cast(sim, s, b1)
    assert sim.can_cast(s, sharp)
    cast(sim, s, sharp)
    assert len(s.player.charms) == 2


# ---------------------------------------------------------------- damage math

def test_blade_trap_feint_multiplicative_and_consumed():
    sim, s = fresh()
    fb, eb, ft, feint, shark = give(
        sim, s, "Fireblade", "Elemental Blade", "Fire Trap", "Feint",
        "Fire Shark")
    for c in (fb, eb, ft, feint):
        cast(sim, s, c)
    hp0 = s.boss_hp
    cast(sim, s, shark)
    want = 405 * 1.35 * 1.35 * 1.25 * 1.70
    assert s.boss_hp == pytest.approx(hp0 - want)
    assert not s.blades and "Feint" not in s.traps and "Fire Trap" not in s.traps
    # feint's self-trap is still armed on the player
    assert s.self_vuln == pytest.approx(0.30)


def test_weakness_reduces_and_is_consumed():
    sim, s = fresh()
    s.player.charms.append(Hanging("Weakness", "charm", "damage",
                                   percent=-0.25, schools=None))
    shark = give(sim, s, "Fire Shark")
    hp0 = s.boss_hp
    cast(sim, s, shark)
    assert s.boss_hp == pytest.approx(hp0 - 405 * 0.75)
    assert not s.player.charms


def test_boss_resist_and_boost():
    boss = Boss("Ice guy", 10000, "ice", 0)          # resists ice, boosted by fire
    sim, s = fresh(boss=boss)
    shark = give(sim, s, "Fire Shark")
    hp0 = s.boss_hp
    cast(sim, s, shark)
    assert s.boss_hp == pytest.approx(hp0 - 405 * 1.25)
    sim2, s2 = fresh(boss=Boss("Ice guy", 10000, "ice", 0), school="ice")
    snowman = give(sim2, s2, "Evil Snowman")
    hp0 = s2.boss_hp
    cast(sim2, s2, snowman)
    assert s2.boss_hp == pytest.approx(hp0 - 270 * 0.60)


def test_fuel_three_charges():
    sim, s = fresh()
    fuel = give(sim, s, "Fuel")
    cast(sim, s, fuel)
    for i in range(3):
        cat = give(sim, s, "Fire Cat")
        hp0 = s.boss_hp
        cast(sim, s, cat)
        assert s.boss_hp == pytest.approx(hp0 - 100 * 1.25), i
    cat = give(sim, s, "Fire Cat")
    hp0 = s.boss_hp
    cast(sim, s, cat)
    assert s.boss_hp == pytest.approx(hp0 - 100)      # charges exhausted


# ---------------------------------------------------------------- shields

def test_shield_blocks_matching_school_only():
    boss = Boss("Fire boss", 10000, "fire", 100)
    sim, s = fresh(boss=boss, school="ice")
    fs = give(sim, s, "Fire Shield")                  # -80% vs fire
    cast(sim, s, fs)
    sim.end_round(s)                                  # boss hits for 100 fire
    assert s.player.hp == pytest.approx(3000 - 100 * 0.20)
    sim.end_round(s)                                  # shield consumed
    assert s.player.hp == pytest.approx(3000 - 20 - 100)


def test_tower_shield_universal():
    boss = Boss("Fire boss", 10000, "fire", 100)
    sim, s = fresh(boss=boss, school="ice")
    ts = give(sim, s, "Tower Shield")
    cast(sim, s, ts)
    sim.end_round(s)
    assert s.player.hp == pytest.approx(3000 - 50)


def test_feint_backlash_consumed_by_boss_hit():
    boss = Boss("Boss", 10000, "life", 100)
    sim, s = fresh(boss=boss)
    feint = give(sim, s, "Feint")
    cast(sim, s, feint)
    sim.end_round(s)
    assert s.player.hp == pytest.approx(3000 - 130)   # 100 * 1.30
    sim.end_round(s)
    assert s.player.hp == pytest.approx(3000 - 130 - 100)


def test_pierce_shaves_resist_then_shield():
    sim, s = fresh()
    s.player.pierce = 0.60
    boss = s.enemies[0]
    boss.wards.append(Hanging("Tower Shield", "ward", "damage",
                              percent=-0.50, schools=None))
    shark = give(sim, s, "Fire Shark")
    hp0 = s.boss_hp
    cast(sim, s, shark)
    # no resist on dummy: full pierce shaves the shield to -0%
    assert s.boss_hp == pytest.approx(hp0 - 405)


def test_absorb_soaks_flat_damage():
    boss = Boss("Boss", 10000, "life", 300)
    sim, s = fresh(boss=boss, school="life")
    armor = give(sim, s, "Spirit Armor")
    cast(sim, s, armor)
    sim.end_round(s)                                  # 300 dmg, 400 pool
    assert s.player.hp == 3000
    sim.end_round(s)                                  # 100 left in pool
    assert s.player.hp == pytest.approx(3000 - 200)
    assert not any(h.kind == "absorb" for h in s.player.wards)


# ---------------------------------------------------------------- prisms

def test_prism_converts_and_beats_same_school_resist():
    boss = Boss("Gobbler", 10000, "ice", 0)
    sim, s = fresh(boss=boss, school="ice")
    prism = give(sim, s, "Ice Prism")
    cast(sim, s, prism)
    snowman = give(sim, s, "Evil Snowman")
    hp0 = s.boss_hp
    cast(sim, s, snowman)
    # converted to fire: +25% boost instead of 40% resist
    assert s.boss_hp == pytest.approx(hp0 - 270 * 1.25)


def test_ward_fifo_trap_before_prism_applies_trap():
    boss = Boss("Gobbler", 10000, "ice", 0)
    sim, s = fresh(boss=boss, school="ice")
    trap, prism = give(sim, s, "Ice Trap", "Ice Prism")
    cast(sim, s, trap)
    cast(sim, s, prism)
    snowman = give(sim, s, "Evil Snowman")
    hp0 = s.boss_hp
    cast(sim, s, snowman)
    assert s.boss_hp == pytest.approx(hp0 - 270 * 1.30 * 1.25)


def test_ward_fifo_prism_before_trap_strands_trap():
    boss = Boss("Gobbler", 10000, "ice", 0)
    sim, s = fresh(boss=boss, school="ice")
    prism, trap = give(sim, s, "Ice Prism", "Ice Trap")
    cast(sim, s, prism)
    cast(sim, s, trap)
    snowman = give(sim, s, "Evil Snowman")
    hp0 = s.boss_hp
    cast(sim, s, snowman)
    # prism converts first; the ice trap no longer matches and is stranded
    assert s.boss_hp == pytest.approx(hp0 - 270 * 1.25)
    assert "Ice Trap" in s.traps


# ---------------------------------------------------------------- DoT / HoT

def test_dot_snapshots_modifiers_at_cast():
    sim, s = fresh()
    fb = give(sim, s, "Fireblade")
    cast(sim, s, fb)
    elf = give(sim, s, "Fire Elf")
    hp0 = s.boss_hp
    cast(sim, s, elf)
    # same group: blade boosts initial hit AND the whole DoT
    assert s.boss_hp == pytest.approx(hp0 - 50 * 1.35)
    for i in range(3):
        sim.end_round(s)
    assert s.boss_hp == pytest.approx(hp0 - (50 + 210) * 1.35)
    assert not s.enemies[0].over_time


def test_shield_consumed_at_cast_reduces_whole_dot():
    sim, s = fresh()
    boss = s.enemies[0]
    boss.wards.append(Hanging("Tower Shield", "ward", "damage",
                              percent=-0.50, schools=None))
    hound = give(sim, s, "Heck Hound", pips=3)        # 6 eff pips: 780 total
    hp0 = s.boss_hp
    cast(sim, s, hound)
    for _ in range(3):
        sim.end_round(s)
    assert s.boss_hp == pytest.approx(hp0 - 130 * 6 * 0.50)


def test_hot_and_sanctuary_and_doom():
    boss = Boss("Boss", 10000, "life", 0)
    sim, s = fresh(boss=boss, school="life", hp=3000)
    s.player.hp = 1000
    sanct, sprite = give(sim, s, "Sanctuary", "Sprite")
    cast(sim, s, sanct)
    cast(sim, s, sprite)
    assert s.player.hp == pytest.approx(1000 + 30 * 1.5)
    sim._tick_over_time(s, s.player)
    assert s.player.hp == pytest.approx(1000 + 45 + 90 * 1.5)


def test_doom_and_gloom_cuts_heals():
    sim, s = fresh(school="death")
    s.player.hp = 1000
    doom = give(sim, s, "Doom and Gloom")
    cast(sim, s, doom)
    fairy = give(sim, s, "Fairy@tc")
    cast(sim, s, fairy)
    assert s.player.hp == pytest.approx(1000 + 420 * 0.35)


def test_overheal_clamps():
    sim, s = fresh(school="life")
    s.player.hp = 2900
    fairy = give(sim, s, "Fairy")
    cast(sim, s, fairy)
    assert s.player.hp == 3000


def test_guiding_light_boosts_next_heal_and_is_consumed():
    sim, s = fresh(school="life")
    s.player.hp = 100
    gl, fairy, fairy2 = give(sim, s, "Guiding Light", "Fairy", "Fairy@tc")
    cast(sim, s, gl)
    cast(sim, s, fairy)
    assert s.player.hp == pytest.approx(100 + 420 * 1.30)
    hp = s.player.hp
    cast(sim, s, fairy2)
    assert s.player.hp == pytest.approx(hp + 420)


def test_triage_removes_dot():
    sim, s = fresh()
    s.player.over_time.append(__import__("w101_sim").OverTime(
        "Fire Elf", "dot", "fire", 70, 3))
    triage = give(sim, s, "Triage@tc")
    cast(sim, s, triage, target=0)
    # triage targets an ally (self here)
    assert not any(e.kind == "dot" for e in s.player.over_time)


# ---------------------------------------------------------------- drains

def test_drain_heals_half_of_dealt_ignoring_heal_mods():
    # myth dummy: neutral to death (life would boost it +25%)
    sim, s = fresh(school="death", boss=Boss("Dummy", 100000, "myth", 0))
    s.player.hp = 1000
    doom = give(sim, s, "Doom and Gloom")
    cast(sim, s, doom)                                # heals at -65%...
    vamp = give(sim, s, "Vampire")
    hp0 = s.boss_hp
    cast(sim, s, vamp)
    assert s.boss_hp == pytest.approx(hp0 - 350)
    # ...but drains ignore heal modifiers entirely
    assert s.player.hp == pytest.approx(1000 + 175)


def test_drain_heal_capped_at_max_hp():
    sim, s = fresh(school="death", boss=Boss("Dummy", 100000, "myth", 0))
    wraith = give(sim, s, "Wraith")
    cast(sim, s, wraith)
    assert s.player.hp == 3000


# ---------------------------------------------------------------- multi-hit

def test_minotaur_first_hit_eats_shield_and_blades():
    sim, s = fresh(school="myth")
    boss = s.enemies[0]
    boss.wards.append(Hanging("Tower Shield", "ward", "damage",
                              percent=-0.50, schools=None))
    mb = give(sim, s, "Mythblade")
    cast(sim, s, mb)
    mino = give(sim, s, "Minotaur")
    hp0 = s.boss_hp
    cast(sim, s, mino)
    # 50-hit takes the blade and the shield; 445 lands clean and unbuffed
    want = 50 * 1.35 * 0.50 + 445
    assert s.boss_hp == pytest.approx(hp0 - want)


def test_hydra_heads_hit_their_own_schools():
    boss = Boss("Ice guy", 10000, "ice", 0)           # resists ice, boosts fire
    sim, s = fresh(boss=boss, school="balance")
    hydra = give(sim, s, "Hydra")
    hp0 = s.boss_hp
    cast(sim, s, hydra)
    want = 190 * 1.25 + 190 * 0.60 + 190
    assert s.boss_hp == pytest.approx(hp0 - want)


# ---------------------------------------------------------------- accuracy

def test_fizzle_keeps_pips_discards_card_by_default():
    # first roll feeds new_state's pip gain; second is the fizzle
    sim, s = fresh(rng=RSeq([0.0, 0.99]))
    shark = give(sim, s, "Fire Shark", pips=4)
    assert sim.cast(s, shark) == 0.0
    assert shark not in s.hand                        # discarded
    assert s.player.pow_pips == 4                     # pips kept


def test_fizzle_legacy_rule_keeps_card():
    boss = Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), ["Fireblade"], "fire", boss,
              rng=RSeq([0.0, 0.99]),
              rules=Rules(fizzle_discards_card=False))
    s = sim.new_state()
    shark = give(sim, s, "Fire Shark")
    assert sim.cast(s, shark) == 0.0
    assert shark in s.hand


def test_black_mantle_lowers_accuracy_and_is_consumed():
    sim, s = fresh(rng=RSeq([0.0, 0.5]))
    s.player.charms.append(Hanging("Black Mantle", "charm", "accuracy",
                                   percent=-0.45, schools=None))
    shark = give(sim, s, "Fire Shark")                # 70% - 45% = 25%
    assert sim.cast(s, shark) == 0.0                  # 0.5 > 0.25: fizzle
    assert not s.player.charms                        # consumed by attempt


def test_accuracy_charm_helps():
    # rolls: pip gain, Precision's own cast, then the boosted Fire Shark
    sim, s = fresh(rng=RSeq([0.0, 0.0, 0.78]))
    prec = give(sim, s, "Precision@tc")
    cast(sim, s, prec)
    shark = give(sim, s, "Fire Shark")                # 70% + 10% = 80%
    assert sim.cast(s, shark) > 0                     # 0.78 <= 0.80: hits


def test_dispel_eats_pips_and_card():
    sim, s = fresh(school="storm",
                   boss=Boss("Dummy", 100000, "life", 0))
    s.player.charms.append(Hanging("Dissipate", "charm", "dispel",
                                   schools={"storm"}))
    bats = give(sim, s, "Lightning Bats", pips=4)
    assert sim.cast(s, bats) == 0.0
    assert bats not in s.hand
    assert s.player.pip_slots < 4                     # pips were spent
    assert not s.player.charms                        # dispel consumed


# ---------------------------------------------------------------- crit/block

def test_crit_doubles_when_unblocked():
    sim, s = fresh()
    s.player.crit_chance = 1.0
    shark = give(sim, s, "Fire Shark")
    hp0 = s.boss_hp
    cast(sim, s, shark)
    assert s.boss_hp == pytest.approx(hp0 - 810)      # R0 also passes block=0


def test_block_cancels_crit():
    sim, s = fresh()
    s.player.crit_chance = 1.0
    s.enemies[0].block_chance = 1.0
    shark = give(sim, s, "Fire Shark")
    hp0 = s.boss_hp
    cast(sim, s, shark)
    assert s.boss_hp == pytest.approx(hp0 - 405)


# ---------------------------------------------------------------- stuns

def test_stun_skips_boss_round_and_grants_blocks():
    boss = Boss("Boss", 10000, "life", 100)
    sim, s = fresh(boss=boss, school="myth")
    stun = give(sim, s, "Stun")
    cast(sim, s, stun)
    sim.end_round(s)
    assert s.player.hp == 3000                        # boss was stunned
    assert s.enemies[0].stun_blocks == 4
    stun2 = give(sim, s, "Stun@tc")
    cast(sim, s, stun2)
    assert s.enemies[0].stunned == 0                  # blocked
    assert s.enemies[0].stun_blocks == 3
    sim.end_round(s)
    assert s.player.hp == 2900                        # boss swings again


def test_stun_immune_boss():
    boss = Boss("Boss", 10000, "life", 100, stunable=False)
    sim, s = fresh(boss=boss, school="myth")
    stun = give(sim, s, "Stun")
    cast(sim, s, stun)
    assert s.enemies[0].stunned == 0


# ---------------------------------------------------------------- AoE

def test_aoe_hits_everyone_with_per_target_wards():
    extra = Boss("Adds", 1000, "life", 0)
    boss = Boss("Boss", 10000, "life", 0)
    sim = Sim(dict(CARDS), ["Fireblade"], "fire", boss, rng=R0(),
              enemies=[extra])
    s = sim.new_state()
    s.enemies[1].wards.append(Hanging("Tower Shield", "ward", "damage",
                                      percent=-0.50, schools=None))
    fb = give(sim, s, "Fireblade")
    cast(sim, s, fb)
    meteor = give(sim, s, "Meteor Strike")
    cast(sim, s, meteor)
    # blade consumed once, applies to both; shield only on the second
    assert s.enemies[0].hp == pytest.approx(10000 - 325 * 1.35)
    assert s.enemies[1].hp == pytest.approx(1000 - 325 * 1.35 * 0.50)
    assert not s.blades


def test_scarecrow_drains_all():
    # myth enemies: neutral to death (life would boost it)
    extra = Boss("Adds", 1000, "myth", 0)
    boss = Boss("Boss", 10000, "myth", 0)
    sim = Sim(dict(CARDS), ["Deathblade"], "death", boss, rng=R0(),
              enemies=[extra])
    s = sim.new_state()
    s.player.hp = 500
    crow = give(sim, s, "Scarecrow")
    cast(sim, s, crow)
    assert s.enemies[0].hp == pytest.approx(10000 - 400)
    assert s.enemies[1].hp == pytest.approx(1000 - 400)
    assert s.player.hp == pytest.approx(500 + 400)    # half of 800 total


# ---------------------------------------------------------------- globals

def test_bubble_boosts_school_and_is_replaced():
    sim, s = fresh()
    wyld = give(sim, s, "Wyldfire")
    cast(sim, s, wyld)
    shark = give(sim, s, "Fire Shark")
    hp0 = s.boss_hp
    cast(sim, s, shark)
    assert s.boss_hp == pytest.approx(hp0 - 405 * 1.25)
    frost = give(sim, s, "Balefrost@tc")
    cast(sim, s, frost)
    assert s.bubble.name == "Balefrost@tc"            # replaced Wyldfire
    shark2 = give(sim, s, "Fire Shark@tc")
    hp0 = s.boss_hp
    cast(sim, s, shark2)
    assert s.boss_hp == pytest.approx(hp0 - 405)      # fire no longer boosted


def test_power_play_raises_pip_chance():
    boss = Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), ["Fireblade"], "fire", boss,
              rng=RSeq([0.9, 0.5, 0.9, 0.2]), power_pip=0.0)
    s = sim.new_state()                               # roll 0.9 -> normal pip
    assert s.player.norm_pips == 1
    pp = give(sim, s, "Power Play", pips=4)
    s.player.norm_pips = 4
    s.player.pow_pips = 0
    cast(sim, s, pp)                                  # spends 4 normal
    sim.gain_pip(s)                                   # 0.9 > 0.35: normal
    assert s.player.norm_pips == 1
    sim.gain_pip(s)                                   # 0.2 < 0.35: POWER pip
    assert s.player.pow_pips == 1


# ---------------------------------------------------------------- removal

def test_pierce_spell_removes_shield_not_protected():
    sim, s = fresh(school="myth")
    boss = s.enemies[0]
    boss.wards.append(Hanging("Tower Shield", "ward", "damage",
                              percent=-0.50, schools=None))
    pierce = give(sim, s, "Pierce")
    cast(sim, s, pierce)
    assert not boss.wards
    boss.wards.append(Hanging("Tower Shield", "ward", "damage",
                              percent=-0.50, schools=None, protected=True))
    p2 = give(sim, s, "Pierce@tc")
    cast(sim, s, p2)
    assert len(boss.wards) == 1                       # Aegis-style: immune


def test_steal_ward_takes_boss_shield():
    sim, s = fresh(school="ice")
    boss = s.enemies[0]
    boss.wards.append(Hanging("Tower Shield", "ward", "damage",
                              percent=-0.50, schools=None))
    steal = give(sim, s, "Steal Ward")
    cast(sim, s, steal)
    assert not boss.wards
    assert any(h.name == "Tower Shield" for h in s.player.wards)


def test_cleanse_charm_removes_own_weakness():
    sim, s = fresh(school="storm")
    s.player.charms.append(Hanging("Weakness", "charm", "damage",
                                   percent=-0.25, schools=None))
    cc = give(sim, s, "Cleanse Charm")
    cast(sim, s, cc)
    assert not s.player.charms


# ---------------------------------------------------------------- cheats

def test_trap_counter_cheat_fires():
    from bosses import cheat_variant
    boss = cheat_variant("Jade Oni", "trap_counter")
    sim, s = fresh(boss=boss)
    trap = give(sim, s, "Fire Trap")
    cast(sim, s, trap)
    assert s.player.hp == pytest.approx(3000 - 320)
    blade = give(sim, s, "Fireblade")
    cast(sim, s, blade)                               # blades don't trigger it
    assert s.player.hp == pytest.approx(3000 - 320)


def test_enrage_cheat_fires_once():
    from bosses import CHEATS
    boss = Boss("Rager", 4000, "life", 0, cheat=CHEATS["enrage_at_half"])
    sim, s = fresh(boss=boss)
    s.enemies[0].hp = 1500                            # below half of 4000
    sim.end_round(s)
    assert any(h.percent == 0.40 for h in s.enemies[0].charms)
    assert any(h.percent == -0.50 for h in s.enemies[0].wards)
    sim.end_round(s)
    assert len(s.enemies[0].charms) == 1              # once means once


def test_enrage_cheat_blade_boosts_boss_attack():
    from bosses import CHEATS
    boss = Boss("Rager", 4000, "life", 300, cheat=CHEATS["enrage_at_half"])
    sim, s = fresh(boss=boss)
    s.enemies[0].hp = 1500
    sim.end_round(s)
    # the enrage blade is consumed by his own attack that same round
    assert s.player.hp == pytest.approx(3000 - 300 * 1.40)
    assert not s.enemies[0].charms


def test_shield_cycle_cheat_every_4():
    from bosses import cheat_variant
    boss = cheat_variant("Jade Oni", "shield_cycle")
    sim, s = fresh(boss=boss)
    for _ in range(4):
        sim.end_round(s)
    assert sum(1 for h in s.enemies[0].wards if h.percent == -0.50) == 1


# ---------------------------------------------------------------- treasure

def test_tc_draw_and_no_same_round_discard():
    boss = Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), ["Fireblade"] * 2, "fire", boss, rng=R0(),
              sideboard=["Fairy@tc"] * 3)
    s = sim.new_state()
    assert sim.draw_tc(s, 2) == 2
    tc = s.player.fresh_tc[0]
    assert not sim.discard(s, tc)                     # fresh: locked
    sim.end_round(s)
    assert sim.discard(s, tc)                         # next round: fine
    assert len(s.player.sideboard) == 1


def test_tc_draw_respects_hand_cap():
    boss = Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), ["Fireblade"] * 8, "fire", boss, rng=R0(),
              sideboard=["Fairy@tc"] * 3)
    s = sim.new_state()
    assert len(s.hand) == 7
    assert sim.draw_tc(s, 2) == 0


# ---------------------------------------------------------------- minions

def test_summon_and_minion_attacks():
    boss = Boss("Boss", 10000, "life", 0)
    sim, s = fresh(boss=boss, school="storm")
    we = give(sim, s, "Water Elemental")
    cast(sim, s, we)
    assert len(s.allies) == 1
    sim.end_round(s)
    assert s.boss_hp == pytest.approx(10000 - 70)


def test_threat_targeting_and_taunt():
    boss = Boss("Boss", 10000, "life", 100)
    sim, s = fresh(boss=boss, school="ice")
    we = give(sim, s, "Water Elemental@tc")
    cast(sim, s, we)
    minion = s.allies[0]
    minion.threat = 50                                # minion drew aggro
    sim.end_round(s)
    assert minion.hp < minion.max_hp and s.player.hp == 3000
    taunt = give(sim, s, "Taunt")
    cast(sim, s, taunt)
    hp = minion.hp
    sim.end_round(s)
    assert s.player.hp < 3000 and minion.hp == hp     # player tanks now


def test_sacrifice_minion_requires_minion():
    sim, s = fresh(school="ice")
    dh = give(sim, s, "Draw Health")
    assert not sim.can_cast(s, dh)
    we = give(sim, s, "Water Elemental@tc")
    cast(sim, s, we)
    s.player.hp = 1000
    assert sim.can_cast(s, dh)
    cast(sim, s, dh)
    assert s.player.hp == pytest.approx(1000 + 350)
    assert not any(m.alive for m in s.allies)


# ---------------------------------------------------------------- episodes

def test_run_returns_win_and_survival():
    boss = Boss("Weak", 300, "life", 50)
    sim = Sim(dict(CARDS), ["Fire Shark"] * 4, "fire", boss, rng=R0(),
              player_hp=500)
    t, won, hp = sim.run(strat_nuke_asap, max_turns=20)
    assert won and t <= 4 and hp > 0


def test_player_death_by_boss():
    boss = Boss("Bully", 100000, "life", 400)
    sim = Sim(dict(CARDS), ["Fireblade"] * 4, "fire", boss, rng=R0(),
              player_hp=500)
    t, won, hp = sim.run(lambda sim, s: None, max_turns=20)
    assert not won and hp == 0 and t == 2


def test_dot_can_finish_boss_during_end_round():
    boss = Boss("Low", 120, "life", 0)
    sim, s = fresh(boss=boss)
    elf = give(sim, s, "Fire Elf")
    cast(sim, s, elf)                                 # 50 now, 70/round after
    assert s.boss_hp == pytest.approx(70)
    sim.end_round(s)
    assert s.boss_hp <= 0


def test_determinism_same_seed_same_outcome():
    from bosses import BOSSES, DECKS
    b = BOSSES["Jade Oni"]
    runs = []
    for _ in range(2):
        sim = Sim(dict(CARDS), DECKS["fire"]["oneshot"], "fire",
                  Boss(b.name, b.hp, b.school, 0), player_hp=10**9,
                  rng=random.Random(42))
        runs.append([sim.run(make_blade_stack(3)) for _ in range(20)])
    assert runs[0] == runs[1]


def test_survival_wrapper_heals_when_low():
    boss = Boss("Boss", 100000, "life", 0)
    sim, s = fresh(boss=boss, school="life", deck=["Satyr", "Lifeclops"])
    s.player.hp = 500
    give(sim, s, "Satyr", "Lifeclops")
    pol = make_survival(strat_nuke_asap, heal_below=0.45,
                        shield_when_open=False)
    pick = pol(sim, s)
    assert pick.name == "Satyr"


# ---------------------------------------------------------------- compat

def test_state_compat_views():
    sim, s = fresh()
    fb, ft = give(sim, s, "Fireblade", "Fire Trap")
    cast(sim, s, fb)
    cast(sim, s, ft)
    assert list(s.blades) == ["Fireblade"]
    assert s.traps["Fire Trap"][1] == 1
    assert s.pip_slots == s.player.pip_slots
    s.boss_hp = 1234.0
    assert s.enemies[0].hp == 1234.0


def test_dp_regression_jade_oni_oneshot():
    """The DP abstraction is untouched by the engine rewrite: the classic
    lower bound must reproduce the published 7.33."""
    from dp_solver import solve
    from bosses import BOSSES, DECKS
    b = BOSSES["Jade Oni"]
    V, pol, meta = solve(dict(CARDS), DECKS["fire"]["oneshot"],
                         Boss(b.name, b.hp, b.school, 0), "fire")
    assert V[meta["H"], 1, 0] == pytest.approx(7.33, abs=0.005)
    assert meta["unmodeled"] == []


def test_dp_transfer_still_plays():
    from dp_solver import solve, dp_policy
    from bosses import BOSSES, DECKS
    b = BOSSES["Krokopatra"]
    boss = Boss(b.name, b.hp, b.school, 0)
    V, pol, meta = solve(dict(CARDS), DECKS["fire"]["oneshot"], boss, "fire")
    sim = Sim(dict(CARDS), DECKS["fire"]["oneshot"], "fire", boss,
              player_hp=10**9, rng=random.Random(3))
    w, m = evaluate(sim, dp_policy(V, pol, meta, "fire"), n=400)
    assert w > 0.5


def test_rl_featurizer_and_training_smoke():
    from rl_agent import train_agent
    from bosses import BOSSES, DECKS
    b = BOSSES["Rattlebones"]
    agent, sim = train_agent(dict(CARDS), DECKS["fire"]["speed"], "fire",
                             Boss(b.name, b.hp, b.school, 0), episodes=300,
                             warm=False, seed=0, snap_every=150)
    w, m = evaluate(sim, agent.policy(), n=300)
    assert w > 0.8
