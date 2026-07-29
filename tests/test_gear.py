"""
Gear/pet stat layer: transcription fidelity, the crit-era safety gate,
school tilts staying inside the published envelope, and the report's
own internal consistency checks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gear
from data_full import LIVE_RULES
from gear import (OFFENSE, DEFENSE, GEAR_LEVELS, TILT, ARCHETYPES,
                  PET_TALENTS, MANIFESTS, gear_stats, gear_hp, gear_mana,
                  loadout, pet_stats, power_pip, gear_power_pip,
                  universal_share)
from player_curves import base_power_pip, school_hp
from rulesets import CRIT_ERA
from w101_sim import Sim, Boss


# ---- transcription -------------------------------------------------

def test_every_row_is_a_sorted_min_median_max():
    for table in (OFFENSE, DEFENSE):
        for lvl, row in table.items():
            for stat in row:
                lo, med, hi = stat
                assert lo <= med <= hi, (lvl, stat)


def test_both_tables_cover_the_same_levels():
    assert sorted(OFFENSE) == sorted(DEFENSE) == GEAR_LEVELS
    assert GEAR_LEVELS[0] == 1 and GEAR_LEVELS[-1] == 120


def test_medians_are_monotone_in_level():
    """No stat may go backwards as the wizard levels — the report's
    envelope is a progression, and a transcription typo would show up
    here before it reached a result."""
    for table, n in ((OFFENSE, 4), (DEFENSE, 3)):
        for idx in range(n):
            vals = [table[l][idx][1] for l in GEAR_LEVELS]
            assert vals == sorted(vals), (idx, vals)


def test_interpolation_is_exact_on_the_rows():
    for lvl in GEAR_LEVELS:
        assert gear._interp(OFFENSE, lvl, 0, "median") == \
            float(OFFENSE[lvl][0][1])


def test_interpolation_is_linear_between_rows():
    a = gear._interp(OFFENSE, 100, 0, "median")
    b = gear._interp(OFFENSE, 105, 0, "median")
    for lvl in (101, 102, 103, 104):
        assert gear._interp(OFFENSE, lvl, 0, "median") == \
            a + (b - a) * (lvl - 100) / 5


def test_level_is_clamped_not_extrapolated():
    """Above 120 the report gives nothing; hold, never invent."""
    assert gear_stats(120, "fire") == gear_stats(200, "fire")
    assert gear_stats(1, "fire") == gear_stats(-5, "fire")


# ---- the crit-era gate (the load-bearing safety property) ----------

def test_no_crit_rating_without_a_rating_resolver():
    """A rating of 397 in a flat-chance era would mean 'always crit'.
    Regression for the worst silent failure this module could cause."""
    assert "crit" not in gear_stats(100, "fire")
    assert "crit" not in gear_stats(100, "fire", rules=LIVE_RULES)
    assert gear_stats(100, "fire", rules=CRIT_ERA)["crit"] > 300
    assert "crit" not in pet_stats("quad-critical", "mega", "fire")
    assert pet_stats("quad-critical", "mega", "fire",
                     rules=CRIT_ERA)["crit"] == 112


def test_block_is_never_invented():
    """The report publishes no block envelope, so nothing emits one
    unless the caller explicitly opts in."""
    assert "block" not in gear_stats(120, "ice", rules=CRIT_ERA)
    st = gear_stats(120, "ice", rules=CRIT_ERA, block_share=0.35)
    assert st["block"] == st["crit"] * 0.35


def test_classic_era_results_are_untouched_by_gear():
    """Every pre-gear number in this repo must survive the module
    existing: with rules=None the block carries no crit at all."""
    st = loadout(120, "storm", rules=None)["player_stats"]
    assert "crit" not in st and "block" not in st


# ---- school tilts stay inside the published envelope ---------------

def test_tilt_never_leaves_the_reports_band():
    for school in TILT:
        for lvl in (30, 60, 100, 120):
            lo = gear_stats(lvl, school, band="min")["damage"][school]
            hi = gear_stats(lvl, school, band="max")["damage"][school]
            got = gear_stats(lvl, school)["damage"][school]
            assert lo - 1e-9 <= got <= hi + 1e-9, (school, lvl)


def test_tilt_direction_matches_the_report():
    """'Storm and Fire skewing farther into damage/critical, Ice/Life
    farther into health/resist, Balance/Death middle-of-the-road.'"""
    dmg = {s: gear_stats(120, s)["damage"][s] for s in TILT}
    assert dmg["storm"] > dmg["fire"] > dmg["balance"] > dmg["life"] > dmg["ice"]
    res = {s: gear_stats(120, s)["resist"]["*"] for s in TILT}
    assert res["ice"] > res["life"] > res["balance"] > res["fire"] > res["storm"]
    assert gear_hp(120, "ice") > gear_hp(120, "storm")
    assert TILT["myth"] == TILT["balance"]          # report doesn't name myth


def test_offense_and_defense_tilts_are_mirrored():
    """Regression: the first cut used one tilt for every stat, which
    handed Storm the top of the RESIST band as well as the damage band
    — a strictly-better Storm rather than a school identity."""
    for a, b in (("storm", "ice"), ("fire", "life")):
        assert gear_stats(120, a)["damage"][a] > gear_stats(120, b)["damage"][b]
        assert gear_stats(120, a)["resist"]["*"] < gear_stats(120, b)["resist"]["*"]
    # A 0.5 tilt is the band MIDPOINT, which is not the published
    # median — the report's envelope is right-skewed at high level, so
    # a middle-of-the-road school lands slightly below its median.
    lo = gear_stats(120, "balance", band="min")["damage"]["balance"]
    hi = gear_stats(120, "balance", band="max")["damage"]["balance"]
    med = gear_stats(120, "balance", band="median")["damage"]["balance"]
    assert gear_stats(120, "balance")["damage"]["balance"] == (lo + hi) / 2
    assert (lo + hi) / 2 < med


def test_neutral_stats_ignore_the_tilt():
    """The report makes no school claim about accuracy or mana, so
    every school sits at the median rather than inheriting the tilt."""
    accs = {gear_stats(120, s).get("accuracy") for s in TILT}
    manas = {gear_mana(120, s) for s in TILT}
    assert len(accs) == 1 and len(manas) == 1


def test_hp_is_base_curve_plus_gear_bonus():
    for school in ("ice", "storm"):
        assert gear_hp(120, school, band="median") == \
            school_hp(school, 120) + DEFENSE[120][1][1]


# ---- the power-pip conflict, kept visible --------------------------

def test_power_pip_conflict_is_resolved_toward_the_base_rule():
    """The report's column reads as gear-only low and as a total high;
    below 50 the documented base rule wins, above it the report does."""
    assert gear_power_pip(20) < base_power_pip(20)
    assert power_pip(20) == base_power_pip(20)
    assert power_pip(60) == gear_power_pip(60) > base_power_pip(60)
    assert power_pip(120) == 1.0


# ---- the damage split ----------------------------------------------

def test_universal_share_tapers_between_its_two_anchors():
    assert universal_share(30) == 10 / 22          # Sky Iron Hasta
    assert universal_share(56) == 0.0              # Wintertusk crafted
    assert universal_share(10) == universal_share(30)     # held below
    assert universal_share(120) == 0.0                    # held above
    assert 0 < universal_share(43) < 10 / 22


def test_endgame_gear_damage_is_school_specific():
    """The asymmetry the archmastery result rides on: an off-school
    splash gets no gear damage at all in the endgame."""
    st = gear_stats(120, "fire")
    assert set(st["damage"]) == {"fire"}
    assert gear_stats(30, "fire")["damage"]["*"] > 0     # Hasta era


# ---- pets: the report's own numbers reconcile ----------------------

def test_triple_double_reconciles_from_talent_ranges():
    """The report quotes ~22% attack / 15% defense for a max-stat
    triple-double. That is Dealer 10 + Giver 6 + Pain-Giver 6 and
    Proof 10 + Defy 5 from the wiki ranges — an independent
    cross-check between two of its sources, not an assumption."""
    st = pet_stats("triple-double", "mega", "fire")
    assert round(sum(st["damage"].values()) * 100) == 22
    assert round(st["resist"]["*"] * 100) == 15


def test_ultra_jewel_matches_the_quoted_deltas():
    """Mighty takes 22/15 to 25/17; Thinking Cap takes 112 to 124."""
    st = pet_stats("triple-double", "ultra", "fire")
    assert round(sum(st["damage"].values()) * 100) == 25
    assert round(st["resist"]["*"] * 100) == 17
    assert pet_stats("quad-critical", "ultra", "fire",
                     rules=CRIT_ERA)["crit"] == 124


def test_quad_crit_prefix_sits_inside_the_reports_age_envelope():
    """The report's pet-age table gives crit windows by age; the
    archetype builder must land inside its own report's envelope."""
    envelope = {"adult": (20, 60), "ancient": (40, 90),
                "epic": (80, 112), "mega": (112, 112)}
    for age, (lo, hi) in envelope.items():
        c = pet_stats("quad-critical", age, "fire", rules=CRIT_ERA)["crit"]
        assert lo <= c <= hi, (age, c)


def test_ultra_manifests_no_sixth_talent():
    assert MANIFESTS["ultra"] == MANIFESTS["mega"] == 5
    assert MANIFESTS["baby"] == 0
    for build in ARCHETYPES.values():
        assert len(build) == 5


def test_baby_pet_contributes_nothing():
    assert pet_stats("triple-double", "baby", "fire") == {}
    assert pet_stats(None, "mega", "fire") == {}


def test_talent_ranges_are_ordered():
    for name, (_, _, lo, hi, mx) in PET_TALENTS.items():
        assert lo <= hi, name
        assert mx is None or mx >= hi, name


def test_school_scoped_talent_needs_a_school():
    try:
        pet_stats("triple-double", "mega")
    except ValueError as e:
        assert "school" in str(e)
    else:
        raise AssertionError("school-scoped talent silently dropped")


# ---- integration ---------------------------------------------------

def test_loadout_merges_gear_and_pet_additively():
    g = gear_stats(120, "fire", rules=CRIT_ERA)
    lo = loadout(120, "fire", rules=CRIT_ERA, pet="triple-double")
    st = lo["player_stats"]
    assert st["damage"]["fire"] == g["damage"]["fire"] + 0.16
    assert round(st["damage"]["*"], 10) == 0.06         # Pain-Giver
    assert round(st["resist"]["*"] - g["resist"]["*"], 10) == 0.15
    assert lo["player_hp"] == gear_hp(120, "fire")      # no Health Gift
    assert lo["power_pip"] == 1.0


def test_loadout_drives_a_real_sim():
    boss = Boss("dummy", 3000, "death", 120)
    boss.resist_map = {}
    boss.boost_map = {}
    sim = Sim(dict(_CARDS), ["Sunbird", "Sunbird", "Fireblade"], "fire",
              boss, rules=CRIT_ERA,
              **loadout(100, "fire", rules=CRIT_ERA, pet="triple-double"))
    s = sim.new_state()
    assert s.player.crit_chance > 300           # a rating, not a chance
    assert s.player.max_hp == gear_hp(100, "fire")
    assert s.player.damage_bonus["fire"] > 1.0  # gear + pet, >100%


def test_mana_is_transcribed_but_inert():
    assert gear_mana(120, "fire") > 0
    assert "mana" not in loadout(120, "fire")["player_stats"]


from data_full import load_spells_full           # noqa: E402
_ROOT = Path(__file__).resolve().parent.parent
_CARDS = load_spells_full(str(_ROOT / "spells_full.json"),
                          str(_ROOT / "cards_clean.json"))
