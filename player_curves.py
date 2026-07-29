"""
School base-stat curves from the July 2026 research report (official-
forum anchors; see docs/RESEARCH.md for provenance and caveats).

Two things this module refuses to fabricate, per the report:
  - HP between the level 1 and 120 anchors is INTERPOLATED linearly
    and tagged so — the report is explicit that linearity is not
    proven ("average gain per level ... does not prove that health
    rises linearly"). Above level 120 the report provides no anchors,
    so the L120 value is held (clamped), not extrapolated.
  - Practical combat stats (damage, resist, accuracy, crit, pierce)
    are gear-dominated and have NO defensible base curve; they are
    deliberately absent here.

The base power-pip rule is universal and well documented: 0% before
level 10, ~1 point per level from 10 to the 40% base cap at 50, flat
thereafter (gear raises the practical value; 0.85 in the endgame
experiments is a geared figure, not a base one).
"""

# (level-1 HP, level-120 HP) — official-forum anchor values
HP_ANCHORS = {
    "storm":   (400, 2343),
    "fire":    (425, 3026),
    "myth":    (425, 3026),
    "death":   (450, 3364),
    "life":    (460, 3714),
    "balance": (480, 3686),
    "ice":     (500, 4204),
}

MAX_LEVEL = 180


def school_hp(school, level):
    """Base HP at `level` (1..180): exact at the anchors, linear
    between them (approximation, documented above), clamped to the
    L120 anchor beyond it."""
    lo, hi = HP_ANCHORS[school]
    lvl = max(1, min(level, MAX_LEVEL))
    if lvl >= 120:
        return hi
    return round(lo + (hi - lo) * (lvl - 1) / 119)


def base_power_pip(level):
    """Universal base power-pip chance: 0 before 10, +1 point per
    level from 10, capped at the 40% base at 50+."""
    if level < 10:
        return 0.0
    return min(0.40, 0.01 * (level - 10))
