"""
Gear and pet stat layer, levels 1-120 (roadmap: the gear/pet layer).

`player_curves.py` deliberately stops at base HP and the base power-pip
rule, with a stated refusal: "Practical combat stats (damage, resist,
accuracy, crit, pierce) are gear-dominated and have NO defensible base
curve; they are deliberately absent here." This module is that missing
layer, supplied by the July 2026 gear/pet research report.

WHAT IS SOURCED VS MODELED — the report is unusually candid about its
own method, and that candour is carried through here rather than
smoothed away:

  GEAR TABLES ARE A MODELED ENVELOPE. The report says so directly: it
  is "not a claim that every item in the database was exhaustively
  scraped", but a benchmark model anchored to the consensus ladder
  (Bazaar to 25, Mount Olympus 30, Wintertusk 56, Waterworks 60,
  Poseidon 70, Tartarus/Hades 90, Darkmoor 100, Polaris 110, Mirage
  Time Warden 120). The min/median/max columns are "what you should
  expect to see", not a distribution over the item universe. They are
  transcribed here verbatim; nothing is invented on top.

  PET TALENT RANGES ARE SOURCED. Wizard101 Central's calculator pages
  expose real per-talent ranges (Pain-Giver +1..+6, Spell-Proof
  +1..+10, Critical Striker +1..+31, and so on), and the maturation
  ladder (one manifest per age Teen->Mega, Ultra being the jewel tier)
  is documented. The archetype TOTALS the report quotes reconcile
  exactly against those ranges, which is a real cross-check: see
  test_gear.py::test_triple_double_reconciles_from_talent_ranges.

Three things this module refuses to fabricate, in the same spirit:

  - BLOCK. The report tracks block as a live stat and names gear that
    carries it, but publishes no envelope. `gear_stats` therefore
    emits no block rating unless a caller passes `block_share`
    explicitly. There is no default.
  - INTER-LEVEL VALUES are linear between the report's 5-level rows,
    tagged as approximation exactly as `school_hp` tags its own.
  - UNIVERSAL VS SCHOOL DAMAGE is split from the only two anchors that
    exist (see UNIVERSAL_SHARE), not from an invented shape.

KNOWN CONFLICT, left visible: the report's power-pip column reads as
gear-only at low level (5% at L20) but as a total character sheet
figure at high level (100% at L120) — 5% cannot be a total when the
documented base rule already gives 10% at L20. `power_pip` resolves it
by taking the max of the two curves and preferring the better-sourced
base rule where they disagree; `gear_power_pip` exposes the report's
raw column so the conflict stays inspectable.
"""
from player_curves import base_power_pip, school_hp

RULES_NOTE = ("gear envelopes MODELED (benchmark method); pet talent "
              "ranges SOURCED from wiki calculator pages")

# ---------------------------------------------------------------------
# The report's two tables, transcribed. Each entry is (min, median, max)
# in the report's own units: damage/resist/accuracy/power-pip in
# PERCENTAGE POINTS, critical in RATING, health/mana flat.
#
# The report also publishes a mean column; it tracks the median within
# 1% at every level and every stat, so only the median is carried.
# ---------------------------------------------------------------------
#      lvl: damage             power_pip          accuracy        critical
OFFENSE = {
    1:   ((0, 0, 0),      (0, 0, 0),      (0, 0, 0),    (0, 0, 0)),
    5:   ((1, 1, 1),      (0, 0, 0),      (0, 0, 0),    (0, 0, 0)),
    10:  ((2, 2, 2),      (0, 0, 0),      (1, 1, 1),    (0, 0, 0)),
    15:  ((3, 3, 3),      (0, 0, 0),      (2, 2, 2),    (0, 0, 0)),
    20:  ((5, 5, 5),      (5, 5, 5),      (3, 3, 3),    (0, 0, 0)),
    25:  ((8, 8, 8),      (12, 12, 12),   (4, 4, 4),    (0, 0, 0)),
    30:  ((21, 22, 23),   (34, 35, 35),   (7, 7, 7),    (0, 0, 0)),
    35:  ((23, 24, 26),   (37, 38, 38),   (8, 8, 8),    (0, 0, 0)),
    40:  ((26, 27, 29),   (41, 42, 43),   (9, 9, 9),    (0, 0, 0)),
    45:  ((28, 31, 32),   (45, 46, 47),   (10, 10, 11), (10, 10, 11)),
    50:  ((32, 35, 37),   (50, 52, 53),   (11, 11, 12), (19, 20, 21)),
    55:  ((39, 43, 46),   (60, 62, 63),   (13, 14, 15), (47, 51, 54)),
    60:  ((48, 53, 57),   (73, 76, 77),   (16, 17, 18), (104, 111, 119)),
    65:  ((52, 57, 62),   (77, 80, 81),   (17, 18, 19), (127, 137, 147)),
    70:  ((56, 63, 68),   (81, 84, 86),   (18, 19, 20), (154, 167, 181)),
    75:  ((61, 68, 74),   (83, 87, 89),   (19, 20, 22), (177, 193, 210)),
    80:  ((65, 73, 80),   (86, 90, 92),   (20, 21, 23), (204, 223, 244)),
    85:  ((69, 78, 86),   (89, 93, 95),   (21, 22, 24), (236, 259, 284)),
    90:  ((76, 87, 96),   (90, 95, 97),   (23, 24, 26), (276, 305, 336)),
    95:  ((82, 94, 105),  (92, 97, 99),   (24, 26, 29), (312, 346, 382)),
    100: ((91, 106, 118), (94, 99, 100),  (26, 28, 31), (356, 397, 441)),
    105: ((96, 112, 125), (95, 100, 100), (28, 30, 33), (391, 438, 488)),
    110: ((103, 120, 135), (94, 100, 100), (30, 32, 36), (435, 489, 547)),
    115: ((109, 128, 144), (94, 100, 100), (32, 34, 38), (474, 535, 601)),
    120: ((114, 135, 153), (94, 100, 100), (33, 36, 40), (513, 581, 656)),
}
#      lvl: resist            health_bonus              mana
DEFENSE = {
    1:   ((0, 0, 0),     (0, 0, 0),                (0, 0, 0)),
    5:   ((0, 0, 0),     (40, 40, 41),             (5, 5, 5)),
    10:  ((0, 0, 0),     (89, 90, 92),             (10, 10, 10)),
    15:  ((0, 0, 0),     (147, 151, 155),          (15, 15, 15)),
    20:  ((1, 1, 1),     (243, 252, 261),          (20, 20, 20)),
    25:  ((2, 2, 2),     (338, 354, 368),          (25, 25, 26)),
    30:  ((6, 6, 6),     (624, 659, 689),          (35, 35, 36)),
    35:  ((7, 7, 7),     (716, 761, 800),          (40, 40, 41)),
    40:  ((8, 8, 8),     (855, 915, 967),          (45, 45, 47)),
    45:  ((9, 9, 10),    (1040, 1120, 1190),       (50, 50, 52)),
    50:  ((10, 10, 11),  (1223, 1326, 1416),       (54, 55, 58)),
    55:  ((12, 13, 14),  (1684, 1839, 1974),       (64, 65, 68)),
    60:  ((18, 19, 21),  (2141, 2353, 2538),       (74, 75, 79)),
    65:  ((19, 20, 22),  (2316, 2561, 2776),       (79, 80, 85)),
    70:  ((21, 22, 24),  (2582, 2873, 3127),       (84, 85, 91)),
    75:  ((22, 23, 25),  (2845, 3185, 3483),       (89, 90, 96)),
    80:  ((23, 24, 27),  (3105, 3498, 3842),       (94, 95, 102)),
    85:  ((23, 25, 28),  (3363, 3812, 4205),       (98, 100, 108)),
    90:  ((25, 27, 30),  (3709, 4230, 4686),       (103, 105, 113)),
    95:  ((27, 29, 33),  (4052, 4649, 5172),       (108, 110, 119)),
    100: ((29, 32, 35),  (4481, 5173, 5778),       (113, 115, 125)),
    105: ((31, 34, 37),  (4818, 5594, 6274),       (118, 120, 131)),
    110: ((32, 36, 40),  (5151, 6016, 6774),       (123, 125, 137)),
    115: ((33, 37, 41),  (5481, 6440, 7279),       (127, 130, 143)),
    120: ((35, 39, 44),  (5808, 6864, 7788),       (132, 135, 148)),
}
GEAR_LEVELS = sorted(OFFENSE)
BANDS = {"min": 0, "median": 1, "max": 2}

# The report's benchmark jumps, kept as data so a probe can test its
# headline claim ("a series of benchmark jumps") instead of asserting it.
BREAKPOINTS = (30, 56, 60, 90, 100, 110, 120)

# School identity, as the report states it: "Storm and Fire skewing
# farther into damage/critical, Ice/Life skewing farther into
# health/resist, and Balance/Death tending more toward middle-of-the-
# road totals." Encoded as a position in [0,1] on the report's own
# min..max band — so a tilt REDISTRIBUTES within the published envelope
# and never leaves it. MYTH is not named by the report, so it sits at
# the middle rather than being guessed into one camp.
TILT = {
    "storm":   1.00,
    "fire":    0.85,
    "myth":    0.50,
    "balance": 0.50,
    "death":   0.50,
    "life":    0.20,
    "ice":     0.00,
}

# Damage split. The report names exactly two items precisely enough to
# anchor this: Sky Iron Hasta at 30 is +10% UNIVERSAL out of a ~22
# median, and the level-56 Wintertusk crafted set is "+46% Myth
# Damage" — school-specific, no universal component. Two anchors, so:
# linear between them, held flat outside, same discipline as school_hp.
# Consequence worth stating: this makes endgame gear damage entirely
# school-specific, which is the documented modern shape and which makes
# the archmastery splash tax an UPPER bound rather than a neutral one.
UNIVERSAL_SHARE = ((30, 10 / 22), (56, 0.0))

# Modeled, opt-in only: the report publishes no block envelope.
BLOCK_SHARE_SUGGESTED = 0.35


def _interp(table, level, idx, band):
    """Report row at `level`, linear between the 5-level rows."""
    b = BANDS[band] if isinstance(band, str) else band
    lvl = max(GEAR_LEVELS[0], min(int(level), GEAR_LEVELS[-1]))
    if lvl in table:
        return float(table[lvl][idx][b])
    lo = max(l for l in GEAR_LEVELS if l < lvl)
    hi = min(l for l in GEAR_LEVELS if l > lvl)
    a, c = float(table[lo][idx][b]), float(table[hi][idx][b])
    return a + (c - a) * (lvl - lo) / (hi - lo)


def _tilted(table, level, idx, school, band, axis="offense"):
    """A stat at the school's position on the report's band.

    `axis` is which end of the band the school's identity pulls it
    toward. An offensive school sits high on damage/crit and LOW on
    resist/health — the same tilt, inverted — which is what makes
    Storm and Ice mirror images rather than Storm simply being better.
    Stats the report makes no school claim about ('neutral': accuracy,
    mana) stay at the median for everyone.

    An explicit `band` overrides the tilt entirely, for querying the
    published envelope itself.
    """
    if band is not None:
        return _interp(table, level, idx, band)
    if axis == "neutral":
        return _interp(table, level, idx, "median")
    pos = TILT[school] if axis == "offense" else 1.0 - TILT[school]
    lo = _interp(table, level, idx, "min")
    hi = _interp(table, level, idx, "max")
    return lo + (hi - lo) * pos


def universal_share(level):
    """Fraction of gear damage that is UNIVERSAL rather than own-school
    (see UNIVERSAL_SHARE for the two anchors this is built from)."""
    (l0, s0), (l1, s1) = UNIVERSAL_SHARE
    if level <= l0:
        return s0
    if level >= l1:
        return s1
    return s0 + (s1 - s0) * (level - l0) / (l1 - l0)


def gear_power_pip(level, band="median"):
    """The report's power-pip column, raw, as a fraction. Conflicts
    with the documented base rule below level ~50 — see module docstring
    and `power_pip`, which resolves it."""
    return _interp(OFFENSE, level, 1, band) / 100.0


def power_pip(level, band="median"):
    """Practical power-pip chance: the better-sourced base rule and the
    report's gear column, whichever is higher at this level."""
    return min(1.0, max(base_power_pip(level), gear_power_pip(level, band)))


def gear_hp(level, school, band=None):
    """Total health: the base school curve plus the gear bonus. Ice and
    Life tilt high on both, which is why the gap widens with level."""
    bonus = _tilted(DEFENSE, level, 1, school, band, "defense")
    return int(round(school_hp(school, level) + bonus))


def gear_mana(level, school, band=None):
    """Carried for completeness and provenance. NOT MODELED by the
    engine — this simulator has no mana resource, so nothing consumes
    it. Present so the transcription is whole, not because it acts."""
    return int(round(_tilted(DEFENSE, level, 2, school, band,
                                "neutral")))


def gear_stats(level, school, band=None, rules=None, block_share=None):
    """Player stat block for `Sim(player_stats=...)` at a gear level.

    CRIT SAFETY, deliberate and load-bearing: `Actor.crit_chance` holds
    a PROBABILITY in the classic era and a RATING only when the ruleset
    installs a rating resolver. Emitting a rating of 397 into a
    classic-era sim would read as "always crit". So the crit and block
    ratings are emitted ONLY when `rules.crit_resolver` is set, and
    `rules=None` means no criticals. This is why every criticals-off
    number in this repo stays bit-identical with gear switched on.
    """
    dmg = _tilted(OFFENSE, level, 0, school, band) / 100.0
    res = _tilted(DEFENSE, level, 0, school, band, "defense") / 100.0
    acc = _tilted(OFFENSE, level, 2, school, band, "neutral") / 100.0
    u = universal_share(level)
    st = {}
    if dmg:
        st["damage"] = {}
        if dmg * (1 - u):
            st["damage"][school] = dmg * (1 - u)
        if dmg * u:
            st["damage"]["*"] = dmg * u
    if res:
        st["resist"] = {"*": res}
    if acc:
        st["accuracy"] = acc
    if rules is not None and getattr(rules, "crit_resolver", None):
        crit = _tilted(OFFENSE, level, 3, school, band)
        if crit:
            st["crit"] = crit
            if block_share is not None:
                st["block"] = crit * block_share
    return st


# ---------------------------------------------------------------------
# Pets. Ranges are SOURCED (wiki calculator pages); which talents show
# up at which age is MODELED — see manifest order below.
# ---------------------------------------------------------------------
AGES = ("baby", "teen", "adult", "ancient", "epic", "mega", "ultra")
# Ultra does NOT manifest a sixth regular talent; it is the jewel tier.
MANIFESTS = {"baby": 0, "teen": 1, "adult": 2, "ancient": 3,
             "epic": 4, "mega": 5, "ultra": 5}

#  name -> (stat family, scope, effect_lo, effect_hi, max_with_selfish)
#  scope "*" = universal, "school" = the pet owner's school.
#  `effect_hi` is the normal full-strength manifestation players treat
#  as finished; the last column is the calculator ceiling once selfish
#  stat-boosting talents are present. None = the report gives no ceiling.
PET_TALENTS = {
    "Pain-Giver":      ("damage", "*", 1, 6, 9),
    "School-Giver":    ("damage", "school", 1, 6, 9),
    "School-Dealer":   ("damage", "school", 1, 10, 14),
    "Spell-Proof":     ("resist", "*", 1, 10, 15),
    "Spell-Defying":   ("resist", "*", 1, 5, None),
    "Sharp-Shot":      ("accuracy", "*", 1, 6, 9),
    "Pip O' Plenty":   ("power_pip", "*", 1, 5, 7),
    "Critical Striker": ("crit", "*", 1, 31, 45),
    "Health Gift":     ("health", "*", 1, 129, 188),
}

# The two end-state builds the report identifies, in MANIFEST ORDER.
# Real hatching is a lottery over a ten-talent pool; modeling the
# lottery is a separate question. These are the SUCCESSFUL hatch — the
# pet that got what it was bred for — so young-age values here are an
# optimistic read of a project pet, not an average one.
#
# quad-critical carries a documented TOTAL (112 at max stats) that does
# NOT decompose as 4 x Critical Striker's 31; the report names the
# total but not the four talents, so the per-talent value is the total
# divided four ways and flagged rather than reverse-engineered into
# specific talent names. Its fifth slot is Spell-Proof — the near
# universal filler — so the comparison against triple-double is five
# talents against five and not four against five.
ARCHETYPES = {
    "triple-double": [("School-Dealer", None), ("School-Giver", None),
                      ("Pain-Giver", None), ("Spell-Proof", None),
                      ("Spell-Defying", None)],
    "quad-critical": [("Critical Striker", 28.0)] * 4 +
                     [("Spell-Proof", None)],
}
# Ultra jewel deltas, quoted directly by the report: a Mighty jewel
# takes the triple-double from 22/15 to 25/17; a Thinking Cap-style
# selfish push takes quad-critical from 112 to 124.
ULTRA_JEWEL = {"triple-double": {"damage": 3.0, "resist": 2.0},
               "quad-critical": {"crit": 12.0}}


def pet_stats(archetype, age="mega", school=None, rules=None):
    """Cumulative pet contribution at a pet age, as a stat block that
    merges with `gear_stats`. Crit obeys the same era gate as gear."""
    if archetype is None or age == "baby":
        return {}
    if archetype not in ARCHETYPES:
        raise KeyError(f"unknown pet archetype {archetype!r}")
    if age not in MANIFESTS:
        raise KeyError(f"unknown pet age {age!r}")
    crit_era = rules is not None and getattr(rules, "crit_resolver", None)
    acc = {}
    for talent, override in ARCHETYPES[archetype][:MANIFESTS[age]]:
        fam, scope, _, hi, _ = PET_TALENTS[talent]
        if scope == "school" and not school:
            raise ValueError(f"{talent} is school-scoped: pass school=")
        key = school if scope == "school" else "*"
        acc.setdefault(fam, {})
        acc[fam][key] = acc[fam].get(key, 0.0) + (
            hi if override is None else override)
    if age == "ultra":
        for fam, delta in ULTRA_JEWEL.get(archetype, {}).items():
            if fam in acc:
                k = next(iter(acc[fam]))
                acc[fam][k] += delta
    st = {}
    if "damage" in acc:
        st["damage"] = {k: v / 100.0 for k, v in acc["damage"].items()}
    if "resist" in acc:
        st["resist"] = {k: v / 100.0 for k, v in acc["resist"].items()}
    if "accuracy" in acc:
        st["accuracy"] = sum(acc["accuracy"].values()) / 100.0
    if "crit" in acc and crit_era:
        st["crit"] = sum(acc["crit"].values())
    return st


def pet_power_pip(archetype, age="mega"):
    """Pip O' Plenty contribution, if the build carries it."""
    if archetype is None or archetype not in ARCHETYPES:
        return 0.0
    total = 0.0
    for talent, override in ARCHETYPES[archetype][:MANIFESTS.get(age, 0)]:
        fam, _, _, hi, _ = PET_TALENTS[talent]
        if fam == "power_pip":
            total += hi if override is None else override
    return total / 100.0


def pet_health(archetype, age="mega"):
    """Health Gift contribution, if the build carries it."""
    if archetype is None or archetype not in ARCHETYPES:
        return 0
    total = 0.0
    for talent, override in ARCHETYPES[archetype][:MANIFESTS.get(age, 0)]:
        fam, _, _, hi, _ = PET_TALENTS[talent]
        if fam == "health":
            total += hi if override is None else override
    return int(round(total))


def _merge(a, b):
    """Additive stat merge: gear + pet, per school key."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in a.items()}
    for k, v in b.items():
        if isinstance(v, dict):
            d = out.setdefault(k, {})
            for kk, vv in v.items():
                d[kk] = d.get(kk, 0.0) + vv
        else:
            out[k] = out.get(k, 0.0) + v
    return out


def loadout(level, school, band=None, rules=None, pet=None,
            pet_age="mega", block_share=None):
    """Everything `Sim` needs for a geared wizard, ready to splat:

        Sim(cards, deck, "fire", boss, **loadout(100, "fire", rules=R))

    Returns player_stats / power_pip / player_hp. Criticals are gated
    on the ruleset (see gear_stats); pass `rules=None` and the whole
    block is criticals-free, so classic-era results are untouched.
    """
    st = _merge(gear_stats(level, school, band, rules, block_share),
                pet_stats(pet, pet_age, school, rules))
    return {"player_stats": st,
            "power_pip": min(1.0, power_pip(level, band or "median")
                             + pet_power_pip(pet, pet_age)),
            "player_hp": gear_hp(level, school, band)
            + pet_health(pet, pet_age)}
