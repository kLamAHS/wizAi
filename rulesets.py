"""
Named rulesets: one frozen era per entry (roadmap item 8).

Every result artifact in this repo is stamped with `Rules.ruleset_id`
so numbers from different eras never mix. Until now only two eras
existed — classic (`w101-pve-classic-0.3`) and the live-data variant
(`w101-pve-live-scrape`), both criticals-off and mastery-free. These
are the post-classic ones the engine already had machinery for but no
era ever switched on.

  CLASSIC    the shipped default: no crits, no mastery
  LIVE       game-data cards + exact damage roll tables
  CRIT_ERA   LIVE + rating-based criticals (post-Celestia)
  MASTERY    LIVE + a mastery amulet (one off-school school gets full
             power-pip value)
  MODERN     LIVE + crits + mastery

Confidence: the CARD and BOSS data behind LIVE are extracted game
files, but the crit CURVES are MODELED (see make_rating_crit — the
formulas are not public, and the 2026 research report says so
explicitly). Mastery's rule, by contrast, is a clean documented
mechanic: the amulet's school pays like your own.

PLAYER_STATS carries era-appropriate gear ratings, also modeled —
they exist so the crit machinery has something to bite on, not as a
claim about any particular gear set.
"""
from dataclasses import replace

from data_full import LIVE_RULES
from w101_sim import Rules, make_rating_crit

CLASSIC = Rules()
LIVE = LIVE_RULES

CRIT_ERA = replace(LIVE, ruleset_id="w101-pve-crit-era", era="celestia+",
                   crit_resolver=make_rating_crit(), crit_multiplier=2.0)

MASTERY = replace(LIVE, ruleset_id="w101-pve-mastery", era="celestia+",
                  mastery_school="storm")

MODERN = replace(LIVE, ruleset_id="w101-pve-modern", era="celestia+",
                 crit_resolver=make_rating_crit(), crit_multiplier=2.0,
                 mastery_school="storm")

ERAS = {"classic": CLASSIC, "live": LIVE, "crit-era": CRIT_ERA,
        "mastery": MASTERY, "modern": MODERN}

# gear-like ratings for the crit eras (MODELED; ratings, not chances —
# make_rating_crit turns them into probabilities). Criticals-off eras
# ignore these entirely.
PLAYER_STATS = {"crit": 180.0, "block": 60.0}
BOSS_STATS = {"crit": 120.0, "block": 140.0}


def stats_for(rules):
    """Player stat block appropriate to a ruleset (empty when the era
    has no criticals, so classic numbers stay bit-identical)."""
    return dict(PLAYER_STATS) if rules.crit_resolver else {}
