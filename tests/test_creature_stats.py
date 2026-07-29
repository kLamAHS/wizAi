"""
Per-creature combat stats from the pre-ban scrape: wiring, the crit-era
gate, and the guarantee that nothing pre-existing moved.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import CREATURE_STATS, load_bosses_full, LIVE_RULES
from rulesets import CRIT_ERA
from w101_sim import Boss

ROOT = Path(__file__).resolve().parent.parent
PATH = str(ROOT / "bosses_clean.json")


def _load(rules=None):
    rep = {}
    bosses, _ = load_bosses_full(PATH, report=rep, rules=rules)
    return bosses, rep


def test_coverage_is_recorded_and_matches_the_documented_rates():
    """Coverage is partial and uneven; a result must be able to say how
    much of its boss table was real rather than defaulted."""
    _, rep = _load()
    cov = rep["coverage"]
    assert set(cov) == set(CREATURE_STATS)
    for key, (_attr, documented) in CREATURE_STATS.items():
        assert abs(cov[key]["frac"] - documented) < 0.01, (key, cov[key])


def test_pierce_is_wired_and_fractional():
    bosses, _ = _load()
    pierced = [b for b in bosses.values() if b.pierce]
    assert len(pierced) > 500
    assert all(0 < b.pierce <= 1.0 for b in pierced)     # a fraction
    assert max(b.pierce for b in pierced) < 1.0


def test_starting_pips_are_wired():
    bosses, _ = _load()
    withpips = [b for b in bosses.values() if b.start_pips]
    assert len(withpips) > 1800
    assert all(1 <= b.start_pips <= 7 for b in withpips)


def test_start_pips_only_reach_pool_driven_bosses():
    """A flat-`dmg` boss spends nothing, so giving it pips would be
    cosmetic; only a living boss with a spell pool opens with a rack."""
    b = Boss("x", 1000, "fire", 100, start_pips=4)
    assert b.to_actor().norm_pips == 0
    b.pool = ["Fire Cat"]
    assert b.to_actor().norm_pips == 4


def test_crit_ratings_only_exist_in_a_rating_era():
    """The load-bearing gate, same as gear.py: crit_chance holds a
    PROBABILITY in classic and a RATING only under a rating resolver,
    so a scraped 1085 in a classic sim would mean 'always crit'."""
    plain, rep = _load()
    assert rep["crit_ratings_emitted"] is False
    assert all(b.crit_chance == 0 and b.block_chance == 0
               for b in plain.values())
    assert all(b.crit_chance == 0 for b in _load(LIVE_RULES)[0].values())
    crit, rep2 = _load(CRIT_ERA)
    assert rep2["crit_ratings_emitted"] is True
    rated = [b for b in crit.values() if b.crit_chance]
    assert len(rated) > 500
    assert max(b.crit_chance for b in rated) > 1000       # a rating


def test_wiring_leaves_the_pre_existing_fields_identical():
    """Every result predating this wiring must survive it: health,
    school, resist, boost and the damage estimate are untouched."""
    plain, _ = _load()
    crit, _ = _load(CRIT_ERA)
    for name, b in plain.items():
        c = crit[name]
        assert (b.hp, b.school, b.dmg) == (c.hp, c.school, c.dmg)
        assert b.resist_map == c.resist_map
        assert b.boost_map == c.boost_map
        assert b.outgoing_bonus == c.outgoing_bonus


def test_block_rating_tracks_tier():
    """Recorded because it shaped a probe: every creature with a block
    rating at or above 570 is endgame-sized, so 'high block' and 'big
    HP pool' are not independent knobs in the real data."""
    crit, _ = _load(CRIT_ERA)
    hi = [b for b in crit.values() if b.block_chance >= 570]
    assert hi and min(b.hp for b in hi) > 13000
