"""
Sun enchantments: the stack-identity split that lets a sharpened blade
sit alongside its plain copy, the +10%, and the deck-slot accounting.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import load_spells_full, LIVE_RULES
from w101_sim import (DMG_ENCHANTS, PCT_ENCHANTS, Boss, Hanging, Sim,
                      enchant_card, enchanted_deck_size,
                      register_enchants)

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def _sim(deck, cards):
    b = Boss("dummy", 10**6, "death", 0)
    b.resist_map = {}
    b.boost_map = {}
    return Sim(cards, deck, "fire", b, player_hp=10**9, rules=LIVE_RULES)


def test_enchant_adds_ten_points_and_its_own_source():
    e = enchant_card(CARDS["Fireblade"], "Sharpen Blade")
    assert abs(e.percent - (CARDS["Fireblade"].percent + 0.10)) < 1e-9
    assert abs(e.ops[0]["percent"]
               - (CARDS["Fireblade"].ops[0]["percent"] + 0.10)) < 1e-9
    assert e.source == "enchant-sharp"
    assert e.stack_key != CARDS["Fireblade"].stack_key


def test_enchanted_and_plain_coexist_and_a_duplicate_is_banked():
    """The mechanic, stated as the test: two DIFFERENT stack keys both
    land and both multiply one hit; the SAME key also lands, but only one
    of the pair fires per strike.

    The second half used to read `len(charms) == 2  # duplicate refused`.
    That was the wrong rule in the wrong place -- the game lets a
    duplicate go up, and dedupes when the hit resolves. Enforcing it at
    placement made three 40% traps read as 2.744x on one strike in live
    play, because the placement guard could not fire there (a live-read
    hanging is named `live:<template id>` and matches no card in hand)
    while the resolution had no guard at all.
    """
    cards = dict(CARDS)
    register_enchants(cards, ["Fireblade"], "Sharpen Blade")
    sim = _sim(["Fireblade", "Fireblade+sharp"], cards)
    s = sim.new_state()
    s.player.charms.clear()
    for nm in ("Fireblade", "Fireblade+sharp"):
        sim.execute_ops(s, s.player, nm, cards[nm].source,
                        cards[nm].ops, "fire")
    assert [h.name for h in s.player.charms] == ["Fireblade",
                                                 "Fireblade+sharp"]

    sim.execute_ops(s, s.player, "Fireblade", cards["Fireblade"].source,
                    cards["Fireblade"].ops, "fire")
    assert len(s.player.charms) == 3          # the duplicate goes up too

    # One strike: the two distinct keys apply, the duplicate does not.
    mult = sim._consume_damage_charms(s, s.player, "fire")
    assert abs(mult - 1.35 * 1.45) < 1e-9
    assert [h.name for h in s.player.charms] == ["Fireblade"]   # banked


def test_traps_stack_the_same_way_on_the_enemy():
    cards = dict(CARDS)
    register_enchants(cards, ["Fire Trap"], "Potent Trap")
    sim = _sim(["Fire Trap", "Fire Trap+potent"], cards)
    s = sim.new_state()
    for nm in ("Fire Trap", "Fire Trap+potent"):
        sim.execute_ops(s, s.player, nm, cards[nm].source,
                        cards[nm].ops, "fire", target_idx=0)
    assert len(s.enemies[0].wards) == 2
    assert sorted(h.percent for h in s.enemies[0].wards) == [0.4, 0.5]


def test_potent_feint_is_eighty_forty():
    """Potent Trap boosts BOTH of Feint's wards: the 70% on the enemy
    and the 30% backlash on the wizard, giving 80/40. An earlier cut
    modeled it as 80/30 on the theory that a trap enchant should not
    make your own punishment worse; that was wrong."""
    e = enchant_card(CARDS["Feint"], "Potent Trap")
    assert [o["percent"] for o in e.ops] == [0.8, 0.4]
    assert [o["tgt"] for o in e.ops] == ["enemy", "self"]


def test_enchant_rejects_the_wrong_card_kind():
    for card, ench in (("Fireblade", "Potent Trap"),
                       ("Fire Trap", "Sharpen Blade"),
                       ("Sunbird", "Sharpen Blade")):
        try:
            enchant_card(CARDS[card], ench)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{ench} accepted {card}")


def test_unknown_enchant_raises():
    try:
        enchant_card(CARDS["Fireblade"], "Bewildering")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown enchant accepted")


def test_deck_slot_accounting_counts_the_enchant_card():
    """An enchanted entry is two physical cards; every fair comparison
    in this repo depends on that being counted."""
    assert enchanted_deck_size(["Fireblade", "Sunbird"]) == 2
    assert enchanted_deck_size(["Fireblade+sharp"]) == 2
    assert enchanted_deck_size(["Fireblade", "Fireblade+sharp"]) == 3
    assert all(len(v) == 3 for v in PCT_ENCHANTS.values())
    assert enchanted_deck_size(["Fire Dragon+colossal"]) == 2


def test_enchanted_card_is_castable_and_deals_more():
    """End-to-end: the derived card survives the real cast path."""
    cards = dict(CARDS)
    register_enchants(cards, ["Fire Trap"], "Potent Trap")
    sim = _sim(["Fire Trap+potent", "Sunbird"], cards)
    s = sim.new_state()
    assert any(c.name == "Fire Trap+potent" for c in s.deck + s.hand)


# ---- compounding: the repo owner's worked examples -------------------
# These are the arithmetic the whole damage model rests on, so they are
# pinned against the numbers he gave rather than against my own.

def _mult_of(percents, kind="ward"):
    """Resolve a stack of hangings through the engine and return the
    multiplier it actually produces."""
    cards = dict(CARDS)
    sim = _sim(["Sunbird"], cards)
    s = sim.new_state()
    tgt = s.enemies[0] if kind == "ward" else s.player
    (tgt.wards if kind == "ward" else tgt.charms).clear()
    for i, pct in enumerate(percents):
        h = Hanging(f"h{i}", kind, "damage", percent=pct,
                    schools=None, source=f"src{i}")
        (tgt.wards if kind == "ward" else tgt.charms).append(h)
    if kind == "ward":
        return sim._ward_pass(s, tgt, "fire", 0.0)[0]
    return sim._consume_damage_charms(s, tgt, "fire")


def test_two_blades_compound_to_eighty_two_percent():
    """'Two 35% blades is an 82% boost (1.35 x 1.35 = 1.82), NOT 70%.'"""
    assert abs(_mult_of([0.35, 0.35], "charm") - 1.8225) < 1e-9


def test_three_feints_compound_to_five_and_a_half_times():
    """'Two 80% feints and a 70% feint seems like 230% boost, but it's
    actually 451% (1.7 x 1.8 x 1.8 = 5.508x)'. Additive stacking would
    give 3.30x, so this test is the difference between a simulator that
    can model hard boss fights and one that cannot."""
    got = _mult_of([0.70, 0.80, 0.80])
    assert abs(got - 5.508) < 1e-9
    assert abs(got - (1 + 0.7 + 0.8 + 0.8)) > 2.0     # not additive


def test_damage_stat_is_additive_not_multiplicative():
    """'going from 150 to 160 damage ... 2.6/2.5 is 1.04, so only 4%
    more damage.' The stat enters as 1 + damage, so a 10-point gain
    late is worth far less than it looks."""
    cards = dict(CARDS)
    sim = _sim(["Sunbird"], cards)
    s = sim.new_state()
    s.player.damage_bonus = {"*": 1.50}
    lo = sim._damage_bonus(s.player, "fire")
    s.player.damage_bonus = {"*": 1.60}
    hi = sim._damage_bonus(s.player, "fire")
    assert (lo, hi) == (2.5, 2.6)
    assert abs(hi / lo - 1.04) < 1e-9


def test_flat_damage_enchant_lands_before_the_multipliers():
    """A Colossal is added to BASE damage, so blades and traps multiply
    it too — which is why flat enchants outrun percentage gear."""
    plain = CARDS["Sunbird"]                 # single-hit: no split
    boosted = enchant_card(plain, "Colossal")
    assert boosted.damage == plain.damage + 275
    hits = [o for o in boosted.ops if o["op"] == "hit"]
    plain_hits = [o for o in plain.ops if o["op"] == "hit"]
    assert hits[0]["amount"] == plain_hits[0]["amount"] + 275


def _damage_ops(card):
    return [o.get("amount", o.get("total")) for o in card.ops
            if o["op"] in ("hit", "drain", "dot")]


def test_flat_enchant_raises_the_total_by_exactly_the_bonus():
    """'the total cumulative damage of the DoT spell increases by 275
    overall' — not 275 per tick, and not 275 on top of each component."""
    for name in ("Fire Dragon", "Avenging Fossil", "Sunbird"):
        plain = CARDS[name]
        e = enchant_card(plain, "Colossal")
        assert abs(sum(_damage_ops(e)) - sum(_damage_ops(plain)) - 275) < 1e-6


def test_flat_enchant_splits_proportionally_toward_the_upfront_hit():
    """'the larger portion of the +275 boost typically goes to the
    initial hit, while the remaining value is split among the trailing
    ticks.' Fire Dragon is 540 hit + 435 DoT, so 275 splits 152/123."""
    plain = CARDS["Fire Dragon"]
    e = enchant_card(plain, "Colossal")
    hit_gain = _damage_ops(e)[0] - _damage_ops(plain)[0]
    dot_gain = _damage_ops(e)[1] - _damage_ops(plain)[1]
    assert hit_gain > dot_gain > 0
    assert abs(hit_gain - 275 * 540 / 975) < 1e-3
    assert abs(dot_gain - 275 * 435 / 975) < 1e-3


def test_per_pip_dots_are_refused_rather_than_guessed():
    """Heck Hound's total is scaled by pips at cast time, so a flat
    bonus cannot be folded into it correctly. Refused loudly instead of
    silently multiplied."""
    try:
        enchant_card(CARDS["Heck Hound"], "Colossal")
    except ValueError as e:
        assert "per-pip" in str(e)
    else:
        raise AssertionError("per-pip spell silently enchanted")


def test_damage_enchants_are_ordered_by_tier():
    vals = list(DMG_ENCHANTS.values())
    assert vals == sorted(vals)
    assert DMG_ENCHANTS["Colossal"] == 275 and DMG_ENCHANTS["Epic"] == 300


def test_treasure_and_item_cards_cannot_be_enchanted():
    """'Only works on normal deck spells (not on item cards or other
    treasure cards).'"""
    from dataclasses import replace as _replace
    for src in ("tc", "item", "pet"):
        card = _replace(CARDS["Fireblade"], source=src)
        try:
            enchant_card(card, "Sharpen Blade")
        except ValueError as e:
            assert "normal deck spells" in str(e)
        else:
            raise AssertionError(f"{src} card accepted an enchant")


def test_one_enchant_per_card():
    """'They do not stack multiple enchants on a single card.'"""
    once = enchant_card(CARDS["Fire Dragon"], "Colossal")
    for second in ("Epic", "Strong"):
        try:
            enchant_card(once, second)
        except ValueError:
            pass
        else:
            raise AssertionError("double enchant accepted")


def test_enchant_keeps_the_original_pip_cost():
    """'Costs 0 pips to cast the enchant itself, but adds the base pip
    cost of the original spell.'"""
    for name, ench in (("Fire Dragon", "Colossal"),
                       ("Fireblade", "Sharpen Blade")):
        assert enchant_card(CARDS[name], ench).pips == CARDS[name].pips


def test_a_per_pip_spell_takes_the_bonus_once_not_per_pip():
    """+300 on Tempest is +300 once, whatever the rack. Folding it into
    the per-pip amount multiplied it by the rack -- Epic Tempest at
    seven pips read 2,660 instead of 860 -- so the bonus rides as its
    own flat companion op in the same group, which the engine lands
    exactly once (`_resolve_ops` scales only `per_pip` ops)."""
    base = CARDS["Tempest"]
    e = enchant_card(base, "Epic")
    per_pip = [o for o in e.ops if o.get("per_pip")]
    flat = [o for o in e.ops if o["op"] == "hit" and not o.get("per_pip")]
    assert per_pip and per_pip[0]["amount"] == base.ops[0]["amount"], \
        "the per-pip amount was inflated by the flat bonus"
    assert len(flat) == 1 and flat[0]["amount"] == 300.0
    assert flat[0]["group"] == per_pip[0]["group"], \
        "the bonus must share the original's charm/ward snapshot"
    # The headline stays PER PIP; a flat bonus does not belong in it.
    assert e.damage == base.damage


def test_the_per_pip_bonus_lands_once_in_the_engine():
    """Seven pips of Epic Tempest: 7 x 80 + 300 = 860, not 7 x 380."""
    import random

    cards = dict(CARDS)
    e = enchant_card(cards["Tempest"], "Epic")
    cards[e.name] = e
    sim = _sim([e.name], cards)
    sim.rng = random.Random(7)
    s = sim.new_state()
    s.player.norm_pips, s.player.pow_pips = 7, 0
    s.player.hand = [e]
    before = s.enemies[0].hp
    sim.cast(s, e, 0)
    dealt = before - s.enemies[0].hp
    assert abs(dealt - 860.0) < 1.0, f"Epic Tempest at 7 pips dealt {dealt}"
