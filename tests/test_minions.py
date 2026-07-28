"""
Encounters: the companions the scrape names, and the honest split
between the ones with real stats and the ones without.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import (MINION_HP_SHARE, encounter, load_bosses_full,
                       load_spells_full, LIVE_RULES)
from w101_sim import Sim

ROOT = Path(__file__).resolve().parent.parent
BOSSES, REGISTRY = load_bosses_full(str(ROOT / "bosses_clean.json"))


def test_minions_and_rank_are_loaded():
    withm = [b for b in BOSSES.values() if b.minions]
    assert len(withm) > 400
    assert all(isinstance(m, str) for b in withm for m in b.minions)
    assert any(b.rank > 0 for b in BOSSES.values())


def test_resolved_companions_carry_real_stats():
    """A quarter of the references name creatures that have their own
    page; those must arrive with scraped stats, not a stand-in."""
    name = next(b.name for b in BOSSES.values()
                if b.minions and any(m in BOSSES for m in b.minions))
    boss, comps = encounter(name, BOSSES)
    real = [c for c in comps if not c.name.endswith("(inferred)")]
    assert real
    for c in real:
        assert c.hp == BOSSES[c.name].hp
        assert c.school == BOSSES[c.name].school


def test_resolved_companions_are_peers_not_underlings():
    """Recorded because it changes what the data means: the companions
    that resolve have the SAME rank as their boss and about its health.
    These are multi-boss fights, not boss-plus-adds."""
    pairs = []
    for b in BOSSES.values():
        for m in (b.minions or []):
            if m in BOSSES and b.rank and BOSSES[m].rank:
                pairs.append((b, BOSSES[m]))
    assert len(pairs) > 100
    same_rank = sum(1 for a, c in pairs if a.rank == c.rank)
    assert same_rank / len(pairs) > 0.6, same_rank / len(pairs)


def test_unresolved_companions_are_synthesised_and_labelled():
    name = next(b.name for b in BOSSES.values()
                if b.minions and all(m not in BOSSES for m in b.minions))
    boss, comps = encounter(name, BOSSES)
    assert comps and all(c.name.endswith("(inferred)") for c in comps)
    for c in comps:
        assert c.hp == max(1, int(boss.hp * MINION_HP_SHARE))
        assert c.rank < boss.rank
        assert c.school == boss.school


def test_synth_can_be_switched_off_entirely():
    """`synth=False` is the arm that uses only stats that were really
    scraped — the honest floor for any result that cares."""
    name = next(b.name for b in BOSSES.values()
                if b.minions and all(m not in BOSSES for m in b.minions))
    _b, comps = encounter(name, BOSSES, synth=False)
    assert comps == []


def test_hp_share_is_a_real_knob():
    name = next(b.name for b in BOSSES.values()
                if b.minions and all(m not in BOSSES for m in b.minions))
    small = encounter(name, BOSSES, hp_share=0.1)[1][0].hp
    big = encounter(name, BOSSES, hp_share=0.7)[1][0].hp
    assert big > small * 5


def test_encounter_drives_a_multi_enemy_sim():
    cards = load_spells_full(str(ROOT / "spells_full.json"),
                             str(ROOT / "cards_clean.json"))
    name = next(b.name for b in BOSSES.values()
                if b.minions and len(b.minions) >= 2)
    boss, comps = encounter(name, BOSSES)
    sim = Sim(dict(cards), ["Sunbird", "Fire Dragon"], "fire", boss,
              player_hp=10**9, rules=LIVE_RULES, enemies=comps)
    s = sim.new_state()
    assert len(s.enemies) == 1 + len(comps)
    assert all(e.alive for e in s.enemies)


def test_bosses_without_company_are_unchanged():
    """Every 1v1 result predating this must survive it."""
    solo = [b for b in BOSSES.values() if not b.minions]
    assert len(solo) > 1000
    for b in solo[:50]:
        assert encounter(b.name, BOSSES)[1] == []
