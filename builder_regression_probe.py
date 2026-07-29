"""
The level-20 builder returns a worse deck than it used to. Measured.

Regenerating `results_living.json` after the blade/trap short-circuit
fix moved the living-Krokopatra ladder around, and one of the moves was
not the fix: the deck underneath it changed. The builder used to hand
level 20 a ten-card deck with a Pixie in it; it now returns eight cards
with no heal and no shield. That matters twice over — the deck is
weaker, and `make_survival` on a deck with neither a heal nor a shield
is the identity wrapper, so the "triage" pilot in that ladder stopped
being a distinct policy at all.

This probe separates the two things it could be:

  SCORING   the screen sees both decks and prefers the worse one
  GENERATION the screen never sees the better one, because the
             candidate templates no longer emit it

and it answers GENERATION. Offer the screen both decks and it ranks the
old one first by eleven points on its own n=250 protocol. Ask the
builder for four times as many candidates and it recovers 0.4 points.
The templates changed under it (the blade guarantee, the enchanted-twin
offers) and the neighborhood they sample no longer contains that mix.

Recorded rather than fixed: the fix is a change to candidate
generation, which would move every built-deck number in the repo, and
that deserves its own diff rather than riding along inside a
regeneration pass.

    python builder_regression_probe.py
"""
import copy
import json

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from player_curves import school_hp, base_power_pip
from deck_builder import build_deck, screen
from w101_sim import Sim, evaluate_paired, make_blade_stack, make_survival

LEVEL = 20
N_REPORT = 3000            # the ladder's own budget
CAPACITY = 14

cards = load_spells_full()
bosses, _ = load_bosses_full()

living = copy.copy(bosses["Krokopatra"])
living.dmg = 0
living.pool = ["Thunder Snake", "Lightning Bats", "Storm Shark",
               "Stormblade", "Storm Shield"]
living.archetype = "hitter"
living.discipline = 0.7

HP, PP = school_hp("death", LEVEL), base_power_pip(LEVEL)

# the deck the builder returned before the template changes, taken from
# the results file as it stood at commit 653e8d9
OLD_DECK = ["Banshee", "Banshee", "Sacrifice", "Ghoul", "Deathblade",
            "Deathblade", "Balanceblade", "Hex", "Curse", "Pixie"]
NEW_DECK = json.load(open("results_living.json", encoding="utf-8")
                     )["levels"][str(LEVEL)]["living"]["deck"]

triage = make_survival(make_blade_stack(3))


def report(deck, n=N_REPORT):
    sim = Sim(dict(cards), deck, "death", living, player_hp=HP,
              power_pip=PP, rules=LIVE_RULES)
    st = evaluate_paired(sim, {"triage": triage,
                               "bs3": make_blade_stack(3)}, n=n)
    return {k: v["win_rate"] for k, v in st.items()}


print(f"level-{LEVEL} death wizard ({HP} HP) vs living Krokopatra\n")
print("== the two decks, at the ladder's own budget "
      f"(n={N_REPORT}) ==", flush=True)
for name, deck in (("old (10, Pixie)", OLD_DECK),
                   ("new ( 8, no heal)", NEW_DECK)):
    r = report(deck)
    print(f"   {name:<18} triage {r['triage']*100:5.1f}%   "
          f"bs3 {r['bs3']*100:5.1f}%"
          + ("   <- same policy: no heal, no shield"
             if abs(r["triage"] - r["bs3"]) < 1e-9 else ""), flush=True)

print("\n== is it SCORING? offer the screen both, its own protocol ==",
      flush=True)
table = screen(cards, [OLD_DECK, NEW_DECK], "death", living, LIVE_RULES,
               player_hp=HP, power_pip=PP)
for (name, _), row in zip((("old", 0), ("new", 0)), table):
    print(f"   {name:<18} screen {row[0]*100:5.1f}%", flush=True)
print("   -> no. the screen prefers the old deck when it can see it.",
      flush=True)

print("\n== is it SAMPLING THINNESS? quadruple the candidate budget ==",
      flush=True)
built = {}
for nc in (60, 240):
    dl, _, _, _ = build_deck(cards, "death", living, LIVE_RULES,
                             n_candidates=nc, top_k=3, seed=12,
                             level=LEVEL, player_hp=HP, power_pip=PP,
                             capacity=CAPACITY, log=None)
    r = report(dl)
    heals = sorted({c for c in dl
                    if cards[c].kind in ("heal", "shield")})
    built[nc] = {"deck": dl, "triage": r["triage"], "heals": heals}
    print(f"   n_candidates={nc:>4}  size {len(dl):>2}  "
          f"triage {r['triage']*100:5.1f}%  heals/shields {heals or '[]'}",
          flush=True)
print("   -> no. 4x the budget buys 0.4 points; the templates moved.",
      flush=True)

json.dump({"level": LEVEL, "player_hp": HP,
           "old_deck": OLD_DECK, "new_deck": NEW_DECK,
           "report_n": N_REPORT,
           "report": {"old": report(OLD_DECK), "new": report(NEW_DECK)},
           "screen_n250": {"old": table[0][0], "new": table[1][0]},
           "budget_sweep": {str(k): {"triage": v["triage"],
                                     "size": len(v["deck"]),
                                     "heals": v["heals"]}
                            for k, v in built.items()}},
          open("results_builder_regression.json", "w", encoding="utf-8"),
          indent=1)
print("\nwrote results_builder_regression.json")
