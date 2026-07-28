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
                          random_boss, build_deck, TRAINED,
                          UNIVERSAL_BUFFS)
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


def test_mastery_widens_the_pool_to_the_amulet_school():
    """A mastery amulet makes the amulet school's TRAINED line packable
    — and only that line, so the pet/reskin whitelist still holds."""
    base = legal_pool(CARDS, "fire")
    wide = legal_pool(CARDS, "fire", mastery="storm")
    assert set(base) < set(wide)
    assert "Kraken" in wide and "Kraken" not in base
    allowed = set(TRAINED["fire"]) | set(TRAINED["storm"]) | set(UNIVERSAL_BUFFS)
    assert set(wide) <= allowed, set(wide) - allowed
    assert "Fire Cat Storm" not in wide            # cross-school reskin
    assert "Catalan" not in wide                   # not a trained spell
    assert "Colossus" not in wide                  # third school


def test_mastery_respects_the_level_gate():
    """The amulet grants access, not levels: a low-level wizard still
    cannot pack the amulet school's high-level spells."""
    lo = legal_pool(CARDS, "fire", level=5, mastery="storm")
    hi = legal_pool(CARDS, "fire", level=50, mastery="storm")
    assert "Kraken" not in lo and "Kraken" in hi
    assert set(lo) < set(hi)


def test_pool_stays_school_scoped_without_mastery():
    """Regression: the mastery widening must be opt-in — the default
    build (and every result predating it) sees one school."""
    for school in ("fire", "ice", "storm"):
        assert legal_pool(CARDS, school) == \
            legal_pool(CARDS, school, mastery=None)


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


def test_pool_is_whitelist_only():
    """Regression for the pet-card / reskin leak: 'Firezilla' (pet) and
    'Skeletal Dragon Fire' (cross-school reskin) are variant='core' with
    null level_restriction, indistinguishable from trained spells by any
    dump field — only the curated whitelist keeps them out."""
    for school in TRAINED:
        pool = legal_pool(CARDS, school)
        allowed = set(TRAINED[school]) | set(UNIVERSAL_BUFFS)
        assert set(pool) <= allowed, set(pool) - allowed
    fire = legal_pool(CARDS, "fire")
    for leak in ("Firezilla", "Skeletal Dragon Fire", "Salamander Fire",
                 "Evil Snowman Fire", "Fire Elf Christmas", "Efreet Sun"):
        assert leak not in fire, leak
    assert "Ice Cat" not in legal_pool(CARDS, "ice")   # mutation


def test_level1_pool_is_the_first_spell():
    """A brand-new wizard owns exactly the level-1 quest spell."""
    assert set(legal_pool(CARDS, "fire", level=1)) == {"Fire Cat"}
    assert set(legal_pool(CARDS, "death", level=1)) == {"Dark Sprite"}


def test_trained_whitelist_resolves_in_data():
    """Every curated name must exist in the dump under that dev name —
    catches silent pool shrinkage when spells_full.json is re-exported."""
    missing = [n for names in TRAINED.values() for n in names
               if n not in CARDS]
    missing += [n for n in UNIVERSAL_BUFFS if n not in CARDS]
    assert not missing, missing


def test_build_deck_terminates_on_tiny_pool():
    """A level-1 pool supports only a handful of distinct decks; the
    candidate loop must cap attempts instead of spinning forever."""
    boss = Boss("dummy", 100, "death", 0)   # Lost-Soul scale: the only
    boss.resist_map = {"death": 0.5}        # fight 3x Fire Cat can win
    dl, w, m, table = build_deck(CARDS, "fire", boss, LIVE_RULES,
                                 n_candidates=60, top_k=1, capacity=10,
                                 seed=1, level=1, log=None)
    assert set(dl) == {"Fire Cat"}
    assert w > 0.9


def test_build_deck_p90_objective():
    """objective='p90' ranks the final pick by the slow tail."""
    boss = Boss("dummy", 400, "death", 0)
    boss.resist_map = {"death": 0.5}
    msgs = []
    dl, w, m, _ = build_deck(CARDS, "death", boss, LIVE_RULES,
                             n_candidates=8, top_k=2, seed=2,
                             objective="p90", log=msgs.append)
    assert any("p90" in s for s in msgs)
    assert 0.0 <= w <= 1.0 and len(dl) > 0


def test_sample_deck_offers_defense_only_when_lethal():
    """Shields enter the template only against a boss that hits back."""
    pool = legal_pool(CARDS, "death")
    lethal = Boss("d", 2000, "life", 200)
    immortal = Boss("d0", 2000, "life", 0)
    kinds_lethal, kinds_immortal = set(), set()
    for i in range(30):
        dl = sample_deck(pool, "death", lethal, random.Random(i))
        kinds_lethal |= {CARDS[n].kind for n in dl}
        dl = sample_deck(pool, "death", immortal, random.Random(i))
        kinds_immortal |= {CARDS[n].kind for n in dl}
    assert "shield" in kinds_lethal or "heal" in kinds_lethal
    assert "shield" not in kinds_immortal


def test_build_deck_survival_arm():
    """player_hp= switches the whole pipeline to the lethal objective."""
    boss = Boss("dummy", 800, "death", 150)
    boss.resist_map = {"death": 0.5}
    msgs = []
    dl, w, m, _ = build_deck(CARDS, "fire", boss, LIVE_RULES,
                             n_candidates=10, top_k=2, seed=4,
                             player_hp=900, log=msgs.append)
    assert any("survival at 900 HP" in s for s in msgs)
    assert 0.0 <= w <= 1.0 and len(dl) > 0


def test_random_boss_shape():
    rng = random.Random(3)
    b = random_boss(rng)
    assert 500 <= b.hp <= 8000
    assert b.school in b.resist_map and 0.2 <= b.resist_map[b.school] <= 0.8


# ---- enchantments in the buildable pool ------------------------------

def test_enchants_are_opt_in_and_bit_identical_when_off():
    """Every deck result predating enchantments must survive them."""
    for lvl in (30, 60, 100, None):
        assert legal_pool(CARDS, "fire", level=lvl) == \
            legal_pool(CARDS, "fire", level=lvl, enchants=False)


def test_enchanted_pool_respects_the_unlock_levels():
    from deck_builder import ENCHANT_UNLOCK, _best_damage_enchant
    # Sun enchants open with Celestia at 51, so all of Dragonspyre is
    # a clean control band.
    lo = legal_pool(CARDS, "fire", level=50, enchants=True)
    assert not any("+" in n for n in lo)          # nothing unlocked yet
    mid = legal_pool(CARDS, "fire", level=51, enchants=True)
    assert "Fireblade+sharp" in mid and "Fire Trap+potent" in mid
    assert _best_damage_enchant(50) is None
    assert _best_damage_enchant(51) == "Strong"
    assert _best_damage_enchant(70) == "Colossal"
    assert _best_damage_enchant(120) == "Epic"
    assert ENCHANT_UNLOCK["Epic"] > ENCHANT_UNLOCK["Colossal"]


def test_only_the_best_flat_enchant_is_offered():
    """Players carry their best, not their whole collection — offering
    every tier would blow up the candidate space for nothing."""
    p = legal_pool(CARDS, "fire", level=120, enchants=True)
    tiers = {n.rsplit("+", 1)[-1] for n in p if "+" in n}
    assert "epic" in tiers
    assert not (tiers & {"strong", "giant", "monstrous", "gargantuan",
                         "colossal"})


def test_legality_counts_real_slots_and_per_spell_copies():
    from w101_sim import enchanted_deck_size
    assert check_legal(["Fireblade"] * 3, 16, 3)
    assert not check_legal(["Fireblade"] * 4, 16, 3)
    # an enchanted copy is the SAME spell competing for the same limit
    assert not check_legal(["Fireblade"] * 2 + ["Fireblade+sharp"] * 2,
                           16, 3)
    assert check_legal(["Fireblade"] * 2 + ["Fireblade+sharp"], 16, 3)
    # and it costs two physical slots
    assert enchanted_deck_size(["Fireblade+sharp"] * 5) == 10
    assert not check_legal(["Fireblade+sharp"] * 5, 9, 9)
    assert check_legal(["Fireblade+sharp"] * 5, 10, 9)


def test_sampled_enchanted_decks_stay_legal():
    pool = legal_pool(CARDS, "fire", level=100, enchants=True)
    boss = Boss("B", 6000, "death", 0)
    rng = random.Random(4)
    saw_enchant = False
    for _ in range(60):
        dl = sample_deck(pool, "fire", boss, rng, capacity=24,
                         copy_limit=3)
        if not dl:
            continue
        assert check_legal(dl, 24, 3), dl
        saw_enchant |= any("+" in n for n in dl)
    assert saw_enchant, "enchanted cards never sampled"
