"""
Rank as the bridge to ordinary enemies: the spell-tier heuristic, the
rank->HP curve, and the cross-validation that motivated trusting them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import statistics as st

from data_full import encounter, load_bosses_full, load_spells_full
from enemy_ranks import (DMG_PER_PIP_BAND, DMG_PER_PIP_TYPICAL,
                         NORMAL_HP_ANCHORS, normal_hp, spell_level)

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def test_spell_tier_follows_the_quoted_heuristic():
    """normal ~ rank*6, elite ~ rank*6+5, boss ~ rank*5."""
    assert spell_level(9, "boss") == 45
    assert spell_level(9, "normal") == 54
    assert spell_level(9, "elite") == 59
    for r in range(1, 20):
        assert spell_level(r, "normal") > spell_level(r, "boss")


def test_spell_tier_is_clamped_not_extrapolated():
    """A rank-25 boss is not casting level-125 spells; no such trained
    spell exists, so the proxy is capped rather than trusted upward."""
    assert spell_level(25, "normal") <= 130
    assert spell_level(0, "boss") == 10 and spell_level(None, "boss") == 10


def test_normal_hp_is_exact_at_the_anchors():
    for rank, hp in NORMAL_HP_ANCHORS.items():
        assert normal_hp(rank) == hp


def test_normal_hp_is_monotone_and_log_interpolated():
    vals = [normal_hp(r) for r in range(1, 17)]
    assert vals == sorted(vals)
    # a linear interpolation between 675 (r5) and 5760 (r16) would put
    # r10 near 2986; log space puts it well below that
    assert normal_hp(10) < 2200
    assert normal_hp(10) > NORMAL_HP_ANCHORS[5]


def test_normal_hp_is_held_outside_the_sample():
    """The report is explicit that high-rank health disperses wildly
    (Dream Belloq, rank 17, 300,000 HP), so the curve is held rather
    than extended through that."""
    top = max(NORMAL_HP_ANCHORS)
    assert normal_hp(top + 5) == normal_hp(top) == NORMAL_HP_ANCHORS[top]


def test_synthetic_minions_use_rank_health_not_a_share_of_the_boss():
    bosses, _ = load_bosses_full(str(ROOT / "bosses_clean.json"))
    name = next(b.name for b in bosses.values()
                if b.minions and b.rank and b.rank > 6
                and all(m not in bosses for m in b.minions))
    boss, comps = encounter(name, bosses)
    assert comps
    for c in comps:
        assert c.hp == normal_hp(c.rank)
        assert c.hp != int(boss.hp * 0.35)
    # the legacy rule stays reachable, because the sensitivity sweep
    # varies it and older results quote it
    legacy = encounter(name, bosses, hp_share=0.35)[1]
    assert legacy[0].hp == int(boss.hp * 0.35)


def test_generated_pools_land_in_the_reported_damage_band():
    """The cross-validation worth keeping. The report observes enemy
    natural attack is FLAT across rank at roughly 85-115 damage per pip.
    This repo builds boss pools from school spell lines — a completely
    independent construction — and lands at a median of 99, flat across
    every rank. Two unrelated methods agreeing is evidence; a
    regression here means one of them drifted."""
    bosses, _ = load_bosses_full(str(ROOT / "bosses_clean.json"),
                                 spell_pools=True, cards=CARDS)
    by_rank = {}
    for b in bosses.values():
        if not b.pool or not b.rank:
            continue
        for n in b.pool:
            c = CARDS.get(n)
            if c is None or c.kind not in ("damage", "drain") or c.x_pips:
                continue
            by_rank.setdefault(b.rank, []).append(c.damage / max(c.pips, 1))
    overall = [x for v in by_rank.values() for x in v]
    lo, hi = DMG_PER_PIP_BAND
    assert lo <= st.median(overall) <= hi, st.median(overall)
    assert abs(st.median(overall) - DMG_PER_PIP_TYPICAL) <= 6
    # and FLAT: every well-sampled rank sits in the same band
    for rank, vals in by_rank.items():
        if len(vals) >= 50:
            assert lo <= st.median(vals) <= hi, (rank, st.median(vals))


def test_rank_spell_mode_agrees_on_damage_density():
    """Switching the spell-tier rule from my world inference to the
    report's rank heuristic must not move damage density — the band is
    flat in rank, so the choice moves pip cost and tempo instead."""
    meds = []
    for mode in ("world", "rank"):
        bosses, _ = load_bosses_full(str(ROOT / "bosses_clean.json"),
                                     spell_pools=True, cards=CARDS,
                                     level_mode=mode)
        vals = [CARDS[n].damage / max(CARDS[n].pips, 1)
                for b in bosses.values() if b.pool for n in b.pool
                if n in CARDS and CARDS[n].kind in ("damage", "drain")
                and not CARDS[n].x_pips]
        meds.append(st.median(vals))
    assert abs(meds[0] - meds[1]) <= 5, meds
