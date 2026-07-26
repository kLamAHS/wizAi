"""
Tests for the full-scrape loaders (spells_full.json / bosses_clean.json):
effect-id mapping spot checks against documented spells, provenance
suffixing, honesty of skips/backfills, boss stat parsing, and the
sub-keyed stacking that Tri Blade's three same-named charms require.
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import (load_spells_full, load_bosses_full, LIVE_DECKS,
                       LIVE_RULES)
from w101_sim import Sim, Boss

ROOT = Path(__file__).resolve().parent.parent
REPORT = {}
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"), report=REPORT)
BREPORT = {}
BOSSES, REGISTRY = load_bosses_full(str(ROOT / "bosses_clean.json"),
                                    report=BREPORT)


class R0(random.Random):
    def random(self):
        return 0.0


def test_load_volume_and_honesty():
    assert len(CARDS) > 4000
    assert REPORT["skipped"]                   # skips are reported, not hidden
    reasons = {w for _, w in REPORT["skipped"]}
    assert any("undecoded" in r for r in reasons)
    assert any("wrapper effect" in r for r in reasons)


def test_effect_mapping_spot_checks():
    fb = CARDS["Fireblade"]
    assert fb.ops == [{"op": "charm", "percent": 0.35, "schools": ["fire"],
                       "tgt": "self", "sub": 0}]
    fuel = CARDS["Fuel"]
    assert fuel.ops[0]["charges"] == 3 and fuel.ops[0]["percent"] == 0.40
    feint = CARDS["Feint"]
    assert [round(o["percent"], 2) for o in feint.ops] == [0.70, 0.30]
    assert [o["tgt"] for o in feint.ops] == ["enemy", "self"]
    frost = CARDS["Frostbite"]
    assert frost.ops[0]["op"] == "hit" and frost.ops[0]["amount"] == 80
    assert frost.ops[1]["op"] == "dot" and frost.ops[1]["total"] == 495
    assert frost.ops[0]["group"] == frost.ops[1]["group"]   # one strike
    mino = CARDS["Minotaur"]
    assert mino.ops[0]["group"] != mino.ops[1]["group"]     # two strikes
    judge = CARDS["Judgement"]
    assert judge.x_pips and judge.ops[0].get("per_pip")
    vamp = CARDS["Vampire"]
    assert vamp.ops[0]["op"] == "drain" and vamp.ops[0]["amount"] == 335
    prism = CARDS["Ice Prism"]
    assert prism.ops[0] == {"op": "prism", "from": "ice", "to": "fire",
                            "tgt": "enemy", "sub": 0}
    resh = CARDS["Reshuffle"]
    assert resh.ops[0]["op"] == "reshuffle"


def test_backfill_uses_hit_avg_not_headline():
    elf = CARDS["Fire Elf"]
    hit = next(o for o in elf.ops if o["op"] == "hit")
    assert hit["amount"] == 50                # not the 260 aggregate
    assert "backfilled-avg" in elf.confidence
    kraken = CARDS["Kraken"]
    assert kraken.ops[0]["amount"] == 550


def test_provenance_suffixes_and_sources():
    tcs = [c for c in CARDS.values() if c.source == "tc"]
    assert tcs and all(c.name.endswith("@tc") for c in tcs)
    assert any(c.source == "item" for c in CARDS.values())


def test_tri_blade_places_three_stacking_charms():
    boss = Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), ["Tri Blade"], "fire", boss, rng=R0(),
              rules=LIVE_RULES)
    s = sim.new_state()
    s.player.pow_pips = 7
    tb = sim.cards["Tri Blade"]
    assert sim.can_cast(s, tb)
    sim.cast(s, tb)
    assert len(s.player.charms) == 3          # sub-keys keep all three
    schools = {tuple(h.schools)[0] for h in s.player.charms}
    assert schools == {"fire", "ice", "storm"}
    # the fire hit consumes only the fire charm
    cat = sim.cards["Fire Cat"]
    s.hand.append(cat)
    hp0 = s.boss_hp
    sim.cast(s, cat)
    # LIVE_RULES samples the 80-120 range; the rigged rng (random()==0)
    # lands on the bottom: 80, boosted by the fire charm only
    assert s.boss_hp == pytest.approx(hp0 - 80 * 1.35)
    assert len(s.player.charms) == 2


def test_live_decks_resolve():
    missing = {n for dd in LIVE_DECKS.values() for dl in dd.values()
               for n in dl if n not in CARDS}
    assert missing == set()


def test_boss_loader_stats():
    assert len(BOSSES) > 1500
    kroko = BOSSES["Krokopatra"]
    assert kroko.hp == 960 and kroko.school == "storm"
    assert kroko.resist_map == {"storm": 0.7}      # school-specific, not '*'
    assert kroko.boost_map == {"myth": 0.4}
    assert kroko.stunable is False
    assert REGISTRY["Krokopatra"]["dmg_confidence"] == "inferred"


def test_boss_incoming_mult_uses_maps():
    kroko = BOSSES["Krokopatra"]
    assert kroko.incoming_mult("storm") == pytest.approx(0.3)
    assert kroko.incoming_mult("myth") == pytest.approx(1.4)
    assert kroko.incoming_mult("fire") == pytest.approx(1.0)


def test_reshuffle_refills_deck_but_not_itself():
    boss = Boss("Dummy", 100000, "life", 0)
    sim = Sim(dict(CARDS), ["Fire Cat"] * 3 + ["Reshuffle"], "fire", boss,
              rng=R0(), rules=LIVE_RULES)
    s = sim.new_state()
    s.player.pow_pips = 7
    for _ in range(3):                        # burn all Fire Cats
        cat = next(c for c in s.hand if c.name == "Fire Cat")
        sim.cast(s, cat)
        s.player.pow_pips = 7
    assert not s.deck
    resh = next(c for c in s.hand if c.name == "Reshuffle")
    sim.cast(s, resh)
    assert len(s.deck) == 3                   # cats back, reshuffle gone
    assert all(c.name == "Fire Cat" for c in s.deck)
    assert [c.name for c in s.player.graveyard] == ["Reshuffle"]


def test_live_applies_to_covers_multischool_buffs():
    from w101_sim import buff_applies
    assert buff_applies(CARDS["Spirit Blade"], "death")
    assert buff_applies(CARDS["Tri Blade"], "storm")
    assert not buff_applies(CARDS["Tri Blade"], "death")
    assert buff_applies(CARDS["Feint"], "myth")          # universal
    assert buff_applies(CARDS["Deathblade"], "death")


def test_dp_transfer_survives_drain_only_hands():
    """Regression: dp_policy's fallback must treat drains as nukes — the
    death deck used to stall forever once only Vampire remained."""
    import copy
    from dp_solver import solve, dp_policy
    from w101_sim import evaluate
    b = copy.copy(BOSSES["Jade Oni"])
    b.dmg = 0
    dl = LIVE_DECKS["death"]["oneshot"]
    V, pol, meta = solve(dict(CARDS), dl, b, "death")
    sim = Sim(dict(CARDS), dl, "death", b, player_hp=10**9,
              rules=LIVE_RULES, rng=random.Random(0))
    w, m = evaluate(sim, dp_policy(V, pol, meta, "death"), n=300)
    assert w > 0.5


def test_effect_id_map_matches_stamped_enum():
    """The wiztype enum names are ground truth: every id in the numeric
    map must carry exactly the name we assumed when inferring it."""
    import json
    sp = json.load(open(ROOT / "spells_full.json", encoding="utf-8"))
    want = {1: "kDamage", 3: "kHeal", 5: "kStealHealth",
            23: "kModifyIncomingDamage", 26: "kModifyIncomingDamageType",
            28: "kModifyOutgoingDamage", 29: "kModifyOutgoingHeal",
            38: "kAbsorbDamage", 40: "kModifyAccuracy", 41: "kDispel",
            68: "kStun", 70: "kReshuffle", 72: "kModifyPips",
            76: "kDamageOverTime", 77: "kHealOverTime",
            0: "kInvalidSpellEffect"}
    seen = {}
    for s in sp:
        for e in s["effects"]:
            if "effect_type" in e and e["effect_type_id"] in want:
                seen.setdefault(e["effect_type_id"], set()).add(
                    e["effect_type"])
    for i, name in want.items():
        assert seen.get(i, {name}) == {name}, (i, seen.get(i))


def test_name_dispatch_decodes_new_ops():
    for name, op in (("Stun Block", "stun_block"),
                     ("Triage", "remove"), ("Shatter", "remove")):
        c = CARDS.get(name)
        if c is not None:
            assert any(o["op"] == op for o in c.ops), (name, c.ops)
    assert any(o["op"] == "remove" for c in CARDS.values()
               for o in c.ops)


def test_damage_ranges_sampled_under_live_rules():
    cat = CARDS["Fire Cat"]
    hit = cat.ops[0]
    assert hit.get("spread") == (80.0, 120.0)
    boss = Boss("Dummy", 10**6, "life", 0)
    sim = Sim(dict(CARDS), ["Fireblade"], "fire", boss,
              rng=random.Random(4), rules=LIVE_RULES)
    s = sim.new_state()
    vals = set()
    for _ in range(30):
        s.player.pow_pips = 7
        s.hand.append(cat)
        hp0 = s.boss_hp
        if sim.cast(s, cat) > 0:
            d = hp0 - s.boss_hp
            assert 80 - 1e-6 <= d <= 120 + 1e-6
            vals.add(round(d, 3))
    assert len(vals) > 5                       # actually varying


def test_damage_ranges_off_under_classic_rules():
    from w101_sim import Rules
    cat = CARDS["Fire Cat"]
    boss = Boss("Dummy", 10**6, "life", 0)
    sim = Sim(dict(CARDS), ["Fireblade"], "fire", boss, rng=R0(),
              rules=Rules())                   # classic: averages only
    s = sim.new_state()
    s.player.pow_pips = 7
    s.hand.append(cat)
    hp0 = s.boss_hp
    sim.cast(s, cat)
    assert hp0 - s.boss_hp == pytest.approx(100.0)


def test_live_fight_end_to_end():
    boss = BOSSES["Lord Nightshade"]
    import copy
    b = copy.copy(boss)
    b.dmg = 0
    from w101_sim import make_blade_stack, evaluate
    sim = Sim(dict(CARDS), LIVE_DECKS["fire"]["oneshot"], "fire", b,
              player_hp=10**9, rules=LIVE_RULES, rng=random.Random(1))
    w, m = evaluate(sim, make_blade_stack(2), n=300)
    assert w > 0.8 and m < 15
