"""
World -> level bands, and what they independently corroborate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deck_builder import ENCHANT_UNLOCK
from gear import BREAKPOINTS
from worlds import (SIDE_CONTENT, WORLDS, arc_for, band, level_for,
                    world_for)


def test_bands_are_contiguous_and_non_overlapping():
    spans = sorted(WORLDS.values())
    assert spans[0][0] == 1
    for (lo, hi, _), (nlo, _, _) in zip(spans, spans[1:]):
        assert lo <= hi
        assert nlo == hi + 1, (hi, nlo)


def test_world_lookup_refuses_to_guess_off_the_table():
    assert world_for(1) == "Wizard City"
    assert world_for(73) == "Avalon"
    assert world_for(130) == "Empyrea"
    assert world_for(131) is None and world_for(0) is None
    assert arc_for(55) == 2 and arc_for(10) == 1 and arc_for(120) == 3


def test_level_for_maps_source_prose_into_the_band():
    assert level_for("Celestia", "start") == 51
    assert level_for("Celestia", "end") == 60
    lo, hi = band("Celestia")
    for where in ("start", "early", "mid", "late", "end"):
        assert lo <= level_for("Celestia", where) <= hi


def test_enchant_unlocks_land_in_the_world_the_source_named():
    """The floors were inferred from world names; each must resolve
    into the world it was attributed to. An earlier cut put Strong at
    48 (Dragonspyre) and Colossal at 68 — Zafaria opens at 61."""
    expected = {"Sharpen Blade": "Celestia", "Potent Trap": "Celestia",
                "Strong": "Celestia", "Giant": "Celestia",
                "Monstrous": "Celestia", "Gargantuan": "Celestia",
                "Colossal": "Zafaria", "Epic": "Polaris"}
    for spell, world in expected.items():
        assert world_for(ENCHANT_UNLOCK[spell]) == world, spell


def test_gear_breakpoints_land_on_world_boundaries():
    """Independent corroboration. gear.BREAKPOINTS came from a
    benchmark ladder written with no reference to these bands, yet six
    of its seven steps sit exactly on a world's last level — benchmark
    gear is a world-completion reward.

    The one exception is 56, and it is the informative one: that is
    Wintertusk crafted gear, and Wintertusk is a SIDE world, which is
    precisely why it does not sit on the main ladder."""
    ends = {hi for _, hi, _ in WORLDS.values()}
    on, off = [b for b in BREAKPOINTS if b in ends], \
        [b for b in BREAKPOINTS if b not in ends]
    assert len(on) == 6, on
    assert off == [56]
    assert SIDE_CONTENT["Grizzleheim / Wintertusk"] < 56


def test_side_content_sits_inside_a_real_band():
    for name, lvl in SIDE_CONTENT.items():
        assert world_for(lvl) is not None, name
