"""
Casting bosses: pool construction, the rank->level fit, and the
guarantee that the flat model stays reachable and default.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boss_pools import (SCHOOL_ARCHETYPE, build_pools, estimate_level,
                        fit_rank_to_level, pool_from_notes,
                        pool_from_school)
from data_full import load_bosses_full, load_spells_full
from deck_builder import TRAINED
from worlds import band

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))
RAW = json.load(open(ROOT / "bosses_clean.json", encoding="utf-8"))
FIT = fit_rank_to_level(RAW)


def test_flat_model_is_still_the_default():
    """Every real-boss result predating this must keep reproducing."""
    bosses, _ = load_bosses_full(str(ROOT / "bosses_clean.json"))
    assert all(b.pool is None for b in bosses.values())
    assert bosses["Krokopatra"].dmg > 0


def test_pools_replace_the_flat_hit_when_enabled():
    bosses, rep = {}, {}
    bosses, _ = load_bosses_full(str(ROOT / "bosses_clean.json"),
                                 report=rep, spell_pools=True,
                                 cards=CARDS)
    withpool = [b for b in bosses.values() if b.pool]
    assert len(withpool) > 1700
    for b in withpool:
        assert b.dmg == 0, b.name          # the pool IS the damage
        assert b.pool and all(n in CARDS for n in b.pool)
        assert b.archetype in ("hitter", "healer", "buffer",
                               "debuffer", "tank")
    assert rep["spell_pools"]["enabled"] is True


def test_rank_fit_is_monotone_and_thin_ranks_are_dropped():
    """The raw medians rise cleanly to rank 17 and then collapse on
    tiny samples — rank 24 reads level 20 off 11 records, which once
    handed a 17,240 HP boss a Dark Sprite."""
    ranks = sorted(FIT)
    vals = [FIT[r] for r in ranks]
    assert vals == sorted(vals), vals
    assert FIT[ranks[0]] <= 20 and FIT[ranks[-1]] >= 120
    loose = fit_rank_to_level(RAW, min_n=1)
    assert len(loose) > len(FIT)           # thin ranks really were cut


def test_high_rank_bosses_are_held_at_the_top_not_extrapolated():
    rec = next(r for r in RAW if r["name"] == "Annoushka")
    lvl, how = estimate_level(rec, FIT)
    assert lvl >= 120 and how == "rank-fit-held"


def test_known_worlds_use_the_band_not_the_fit():
    for rec in RAW:
        if rec.get("world") == "Celestia":
            lvl, how = estimate_level(rec, FIT)
            assert how == "world" and lvl == band("Celestia")[1]
            break


def test_spell_notes_yield_real_card_names():
    rec = next(r for r in RAW if r["name"] == "Broken Branch Queen")
    got = pool_from_notes(rec, CARDS)
    assert "Storm Lord" in got
    assert all(n in CARDS for n in got)


def test_precompiled_patterns_change_nothing_but_the_clock():
    """The pattern cache exists because 765k uncached re.search calls
    were 93% of a 29-second build_pools; the pools must come out
    identical either way."""
    from boss_pools import _note_patterns

    pats = _note_patterns(CARDS)
    noted = [r for r in RAW if r.get("spell_notes")][:25]
    noted.append(next(r for r in RAW if r["name"] == "Broken Branch Queen"))
    for rec in noted:
        assert pool_from_notes(rec, CARDS) == \
            pool_from_notes(rec, CARDS, _patterns=pats), rec["name"]


def test_school_pool_respects_the_level_gate():
    lo = pool_from_school("storm", 20, CARDS, TRAINED)
    hi = pool_from_school("storm", 120, CARDS, TRAINED)
    assert lo and hi and lo != hi
    assert "Thunder Snake" in lo
    assert any(CARDS[n].damage > CARDS[x].damage for n in hi for x in lo)
    assert all(n in CARDS for n in lo + hi)


def test_every_pool_has_something_to_cast():
    pools = build_pools(RAW, CARDS, TRAINED)
    assert len(pools) > 1700
    for name, spec in pools.items():
        assert spec["pool"], name
        assert spec["archetype"] in set(SCHOOL_ARCHETYPE.values())


def test_a_casting_boss_actually_casts():
    """End to end: boss damage must route through the cast path, so it
    meets the player's shields and the fizzle roll like any other spell."""
    import random

    from data_full import LIVE_RULES
    from w101_sim import Sim
    bosses, _ = load_bosses_full(str(ROOT / "bosses_clean.json"),
                                 spell_pools=True, cards=CARDS)
    boss = bosses["Krokopatra"]
    sim = Sim(dict(CARDS), ["Tower Shield"] * 6, "fire", boss,
              player_hp=10**6, rules=LIVE_RULES, rng=random.Random(0),
              log_events=True)
    sim.run(lambda s, st: None, max_turns=25)
    ev = sim.last_state.events
    casts = [e for e in ev if e.get("type") == "enemy_cast"]
    assert casts, "casting boss never cast"
    assert {e["card"] for e in casts} <= set(boss.pool)
    # the point of routing damage through the cast path: it can miss,
    # and the boss buffs itself like any caster
    assert any(e.get("type") == "enemy_fizzle" for e in ev)
    assert any(e.get("type") == "charm_consumed"
               and e.get("actor") == "Krokopatra" for e in ev)
