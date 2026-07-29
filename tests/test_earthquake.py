"""
Earthquake: an AoE that wipes the board AFTER it lands.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import LIVE_RULES, load_spells_full
from w101_sim import Boss, Hanging, Sim

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def _sim(deck):
    b = Boss("dummy", 10**6, "death", 0)
    b.resist_map = {}
    b.boost_map = {}
    return Sim(dict(CARDS), deck, "myth", b, player_hp=10**6,
               rules=LIVE_RULES, rng=random.Random(0))


def test_earthquake_targets_the_enemy_side_not_the_caster():
    """The extracted card removed the CASTER's own weaknesses and
    shields — the opposite of the spell — because target 4 (all
    enemies) had no case and fell through to the self/negative
    default."""
    ops = CARDS["Earthquake"].ops
    assert [o.get("what") for o in ops if o["op"] == "remove"] == \
        ["charm_all", "ward_all"]
    assert all(o["tgt"] == "enemies" for o in ops if o["op"] == "remove")


def test_it_wipes_positive_and_negative_alike():
    sim = _sim(["Earthquake"])
    s = sim.new_state()
    foe = s.enemies[0]
    foe.charms.append(Hanging("Mythblade", "charm", "damage", percent=0.35,
                              source="deck"))
    foe.charms.append(Hanging("Weakness", "charm", "damage", percent=-0.25,
                              source="deck"))
    foe.wards.append(Hanging("Tower Shield", "ward", "damage",
                             percent=-0.5, source="deck"))
    foe.wards.append(Hanging("Feint", "ward", "damage", percent=0.7,
                             source="deck"))
    sim.execute_ops(s, s.player, "Earthquake", "deck",
                    CARDS["Earthquake"].ops, "myth")
    assert foe.charms == [] and foe.wards == []


def test_protected_hangings_survive_the_wipe():
    """Aegis/Indemnity semantics: `protected` exists for exactly this."""
    sim = _sim(["Earthquake"])
    s = sim.new_state()
    foe = s.enemies[0]
    # a protected CHARM, not a shield: a shield is consumed by
    # Earthquake's own damage on the way through, which is correct and
    # would hide what this test is checking
    foe.charms.append(Hanging("Aegis Blade", "charm", "damage",
                              percent=0.35, source="deck", protected=True))
    foe.charms.append(Hanging("Mythblade", "charm", "damage", percent=0.35,
                              source="deck"))
    sim.execute_ops(s, s.player, "Earthquake", "deck",
                    CARDS["Earthquake"].ops, "myth")
    assert [h.name for h in foe.charms] == ["Aegis Blade"]


def test_damage_lands_before_the_board_is_stripped():
    """Order of operations: it "calculates and applies the damage first
    (including factoring in any existing blades/traps before they are
    consumed/wiped), and then strips the board". A trap on the target
    must therefore still boost the Earthquake that removes it."""
    hits = []
    for trap in (False, True):
        sim = _sim(["Earthquake"])
        s = sim.new_state()
        foe = s.enemies[0]
        if trap:
            foe.wards.append(Hanging("Myth Trap", "ward", "damage",
                                     percent=0.30, schools={"myth"},
                                     source="deck"))
        before = foe.hp
        sim.execute_ops(s, s.player, "Earthquake", "deck",
                        CARDS["Earthquake"].ops, "myth")
        hits.append(before - foe.hp)
        assert foe.wards == []
    assert hits[1] > hits[0] * 1.25, hits
