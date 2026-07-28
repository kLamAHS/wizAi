"""
Rank as the bridge to ordinary enemies, instead of scraping them all.

The repo owner's proposal, and it is the right one: rather than pull a
page for every mob in the game, map RANK to the boss it fights beside
and derive the mob from that. The rank taxonomy is published, every
creature record already carries a rank, and the community has a
long-standing heuristic for what a rank implies.

WHAT THIS REPLACES. Two of the most arbitrary numbers in the repo:

  MINION_HP_SHARE = 0.35   a stand-in mob's health as a fraction of its
                           boss's — invented, and the sensitivity sweep
                           showed the result swings 53 points across
                           plausible values of it
  world-derived spell tier what a creature casts, inferred from the
                           world its page lists

SPELL-TIER HEURISTIC (community, quoted by the report):

    normal  ~ rank * 6        elite ~ rank * 6 + 5      boss ~ rank * 5

The report is careful that this predicts SPELL ACCESS, not player level
and not difficulty — "useful as a spell-access proxy, but not a
complete predictor of HP, cheats, or endgame encounter difficulty" —
and that is exactly the job it is used for here.

HP BY RANK for non-boss enemies, from the report's normalized sample.
Sparse (four anchors over sixteen ranks) and interpolated in LOG space
because the spread is two orders of magnitude. Same discipline as
`player_curves.school_hp`: exact at the anchors, interpolated between,
clamped outside rather than extrapolated.

WHAT THE REPORT SAYS NOT TO DO. Rank stops predicting difficulty in
late content: Dream Belloq is rank 17 with 300,000 HP against a
sample median in the low thousands, and the Wallaru judges are
dangerous through pierce and outgoing boost rather than health. So
rank is used here for SPELL TIER and for MOB health only — never to
overwrite a boss's scraped health, and never as a difficulty score.
"""
import math

# --- spell-tier proxy -------------------------------------------------
SPELL_TIER = {"normal": lambda r: r * 6,
              "elite": lambda r: r * 6 + 5,
              "boss": lambda r: r * 5}


def spell_level(rank, kind="boss"):
    """Level of spell this creature should have access to. Clamped to
    the trained lines' range; a rank-25 boss is not casting level-125
    spells because no such trained spell exists."""
    if not rank or rank <= 0:
        return 10
    return max(1, min(130, int(SPELL_TIER[kind](rank))))


# --- health by rank, non-boss ----------------------------------------
# rank -> HP, from the report's normalized sample:
#   r1  Lost Soul 65, Wooden Construct 95, Dark Fairy 115 (elite)
#   r2  Gobbler Gorger 285 (elite)
#   r5  Gloom Fairy 675 (elite)
#   r16 Belloq Image 5760 (normal)
NORMAL_HP_ANCHORS = {1: 90, 2: 285, 5: 675, 16: 5760}
_RANKS = sorted(NORMAL_HP_ANCHORS)


def normal_hp(rank):
    """Health of an ordinary (non-boss) enemy of this rank.

    Log-linear between the anchors — the sample spans 90 to 5760, so a
    straight line in linear space would sit far below every midpoint.
    Held flat outside the anchor range: the report is explicit that
    high-rank health disperses wildly (Dream Belloq, rank 17, 300,000)
    and extrapolating a curve through that would be inventing.
    """
    if not rank or rank <= 0:
        return NORMAL_HP_ANCHORS[_RANKS[0]]
    r = max(_RANKS[0], min(int(rank), _RANKS[-1]))
    if r in NORMAL_HP_ANCHORS:
        return NORMAL_HP_ANCHORS[r]
    lo = max(k for k in _RANKS if k < r)
    hi = min(k for k in _RANKS if k > r)
    a, b = (math.log(NORMAL_HP_ANCHORS[lo]),
            math.log(NORMAL_HP_ANCHORS[hi]))
    return int(round(math.exp(a + (b - a) * (r - lo) / (hi - lo))))


# --- the damage-per-pip band -----------------------------------------
# The report's clearest cross-check: enemy natural attack is FLAT across
# rank, "roughly 85-105 damage per pip", reaching 115 at rank 16 — late
# difficulty moves into cheats, pips, pierce and resist instead of
# attack inflation. This repo's boss pools are built from school spell
# lines, an entirely independent construction, and land at a median of
# 99 per pip flat across every rank. Two unrelated methods agreeing is
# worth a regression test; see test_enemy_ranks.py.
DMG_PER_PIP_BAND = (85, 115)
DMG_PER_PIP_TYPICAL = 99
