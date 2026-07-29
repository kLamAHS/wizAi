"""
World -> level bands, as supplied by the repo owner.

Small table, but it does real work in three places:

  1. It dates content. Sources describe spells and gear by WORLD
     ("a Zafaria upgrade", "early Celestia"), and every level floor in
     this repo that was inferred from a world name now resolves through
     one table instead of a scattering of guesses.
  2. It cross-checks the gear report. `gear.BREAKPOINTS` came from a
     benchmark ladder written without reference to these bands, and
     every one of them lands on a world BOUNDARY — see
     test_worlds.py::test_gear_breakpoints_land_on_world_boundaries.
     That is independent corroboration that the benchmark gear sets
     are world-completion rewards.
  3. It gives progression results a readable axis: "level 73" means
     little, "Avalon" means something.

Arc 3 continues past Empyrea in the live game; the table stops where
the owner's list stops rather than extrapolating.
"""

#  name -> (first level, last level, arc)
WORLDS = {
    "Wizard City":  (1, 10, 1),
    "Krokotopia":   (11, 20, 1),
    "Marleybone":   (21, 30, 1),
    "MooShu":       (31, 40, 1),
    "Dragonspyre":  (41, 50, 1),
    "Celestia":     (51, 60, 2),
    "Zafaria":      (61, 70, 2),
    "Avalon":       (71, 80, 2),
    "Azteca":       (81, 90, 2),
    "Khrysalis":    (91, 100, 2),
    "Polaris":      (101, 110, 3),
    "Mirage":       (111, 120, 3),
    "Empyrea":      (121, 130, 3),
}

# Optional detours that sit outside the main ladder. Kept separate
# because they are not level bands — they are "worth doing around here".
SIDE_CONTENT = {
    "Grizzleheim / Wintertusk": 40,   # after Dragonspyre, for XP
    "Mount Olympus (Aquila)": 30,     # the Sky Iron Hasta run
    "Wysteria": 25,                   # unlocks at 25 (2026 rank report)
}

# Worlds past Empyrea that the owner's main table does not band. Only
# what a source actually states is recorded — Novus is tied to level
# 160+ gear and Wallaru is presented as the end of the fourth arc — and
# the rest deliberately stay absent rather than being interpolated into
# existence. Nothing keys off this yet; it exists so the next person
# does not re-derive it from a rank fit.
BEYOND_TABLE = {
    "Novus": "level 160+ gear tier",
    "Wallaru": "end of the fourth story arc, 170+ endgame content",
}

MAX_BANDED_LEVEL = max(hi for _, hi, _ in WORLDS.values())


def world_for(level):
    """The main-line world a wizard of this level is questing. Returns
    None below 1 or above the last banded level rather than guessing."""
    for name, (lo, hi, _arc) in WORLDS.items():
        if lo <= level <= hi:
            return name
    return None


def arc_for(level):
    name = world_for(level)
    return WORLDS[name][2] if name else None


def band(world):
    """(first, last) levels of a world."""
    lo, hi, _ = WORLDS[world]
    return lo, hi


def level_for(world, where="start"):
    """A level inside a world, for turning a source's prose into a
    number. `where` is 'start' | 'early' | 'mid' | 'late' | 'end' —
    the vocabulary sources actually use ("early Celestia")."""
    lo, hi = band(world)
    span = hi - lo
    return {"start": lo,
            "early": lo,
            "mid": lo + span // 2,
            "late": lo + (3 * span) // 4,
            "end": hi}[where]
