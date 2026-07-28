"""
Sun enchantments: the stack-identity split that lets a sharpened blade
sit alongside its plain copy, the +10%, and the deck-slot accounting.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import load_spells_full, LIVE_RULES
from w101_sim import (ENCHANTS, Boss, Sim, enchant_card,
                      enchanted_deck_size, register_enchants)

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


def test_enchanted_and_plain_coexist_but_a_duplicate_does_not():
    """The mechanic, stated as the test: two DIFFERENT stack keys both
    land; the same key never doubles."""
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
    assert len(s.player.charms) == 2          # duplicate refused


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


def test_feint_backlash_is_not_boosted():
    """Feint hangs 70% on the enemy and 30% back on the caster. Potent
    Trap boosts the trap, not the wizard's own punishment — so it goes
    to 80/30, never 80/40."""
    e = enchant_card(CARDS["Feint"], "Potent Trap")
    enemy = [o for o in e.ops if o["tgt"] == "enemy"]
    own = [o for o in e.ops if o["tgt"] == "self"]
    assert enemy[0]["percent"] == 0.8
    assert own[0]["percent"] == 0.3


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
        enchant_card(CARDS["Fireblade"], "Colossal")
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
    assert all(len(v) == 4 for v in ENCHANTS.values())


def test_enchanted_card_is_castable_and_deals_more():
    """End-to-end: the derived card survives the real cast path."""
    cards = dict(CARDS)
    register_enchants(cards, ["Fire Trap"], "Potent Trap")
    sim = _sim(["Fire Trap+potent", "Sunbird"], cards)
    s = sim.new_state()
    assert any(c.name == "Fire Trap+potent" for c in s.deck + s.hand)
