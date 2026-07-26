"""
Deck-construction search: legality, determinism, and prism adaptation.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter

from data_full import load_spells_full, LIVE_RULES
from deck_builder import (legal_pool, sample_deck, check_legal, screen,
                          random_boss, build_deck)
from w101_sim import Boss

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def test_legal_pool_scoped_to_school_plus_universals():
    pool = legal_pool(CARDS, "ice")
    assert "Colossus" in pool and "Ice Prism" in pool
    assert "Tri Blade" in pool                    # cross-trained universal
    assert "Helephant" not in pool                # other school
    assert all(c.source == "deck" for c in pool.values())


def test_sampled_decks_are_legal():
    pool = legal_pool(CARDS, "death")
    boss = Boss("B", 4000, "life", 0)
    rng = random.Random(0)
    for _ in range(60):
        dl = sample_deck(pool, "death", boss, rng, capacity=14, copy_limit=2)
        assert dl and check_legal(dl, 14, 2)
        counts = Counter(dl)
        assert len(dl) <= 14 and max(counts.values()) <= 2
        assert any(pool[n].kind in ("damage", "drain") for n in dl)


def test_prism_only_offered_into_resist():
    pool = legal_pool(CARDS, "ice")
    rng = random.Random(1)
    ice_wall = Boss("Wall", 4000, "ice", 0)       # resists ice
    neutral = Boss("Soft", 4000, "myth", 0)       # doesn't
    walls = sum("Ice Prism" in sample_deck(pool, "ice", ice_wall, rng)
                for _ in range(40))
    softs = sum("Ice Prism" in sample_deck(pool, "ice", neutral, rng)
                for _ in range(40))
    assert walls > 20 and softs == 0


def test_screen_is_deterministic_and_ranks_sanely():
    pool = legal_pool(CARDS, "fire")
    boss = Boss("B", 2000, "life", 0)
    good = ["Helephant"] * 3 + ["Fireblade"] * 2 + ["Fire Trap"] * 2
    bad = ["Fire Cat"] * 3                        # cannot cover 2000 HP well
    a = screen(CARDS, [good, bad], "fire", boss, LIVE_RULES, n=120)
    b = screen(CARDS, [good, bad], "fire", boss, LIVE_RULES, n=120)
    assert a == b
    assert a[0][2] == good                        # ranked first


def test_pool_excludes_boss_only_and_internal_spells():
    """Regression for the reward hack the first search found: the dump
    labels encounter-scripted spells as core; none may be deck-buildable."""
    for school in ("death", "fire", "ice"):
        pool = legal_pool(CARDS, school)
        assert not any(" - " in n or n.startswith("NA ") for n in pool)
        for n, c in pool.items():
            if c.kind in ("damage", "drain"):
                per_pip = c.damage if c.x_pips else c.damage / max(c.pips, 1)
                assert per_pip <= 200, n
    dpool = legal_pool(CARDS, "death")
    assert "Wraith" in dpool and "Feint" in dpool
    assert "Scald - KRBoss Death" not in dpool


def test_level_gated_pool_progression():
    lo = legal_pool(CARDS, "fire", level=5)
    mid = legal_pool(CARDS, "fire", level=30)
    hi = legal_pool(CARDS, "fire", level=50)
    assert set(lo) <= set(mid) <= set(hi)
    assert "Feint" not in lo and "Feint" in mid       # unlocks at 30
    assert "Helephant" not in mid and "Helephant" in hi   # 50
    assert "Fire Cat" in lo                           # null = level 1


def test_random_boss_shape():
    rng = random.Random(3)
    b = random_boss(rng)
    assert 500 <= b.hp <= 8000
    assert b.school in b.resist_map and 0.2 <= b.resist_map[b.school] <= 0.8
