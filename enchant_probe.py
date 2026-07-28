"""
The missing player damage: Feint, sharpened blades, potent traps.

creature_stats.py reported that a solo level-100 single-target fire
deck cannot clear the high-block endgame bosses — 12,027 of Annoushka's
17,240 HP in 80 turns, even with Reshuffle. The repo owner identified
the gap, and it is not in the boss model at all: real wizards cross-
train Feint from Death (70% trap on the enemy, 30% back on themselves)
and carry Sun enchantments that both add +10% AND give the enchanted
card its own stack identity, so a sharpened blade sits alongside the
plain blade instead of replacing it.

Feint was already modeled, both halves. The enchantments were not —
they are absent from the extracted dump because they modify a card in
HAND rather than producing a battle effect, so the parser never saw
one.

THE COMPARISON IS AT MATCHED REAL DECK SLOTS, which is the only way
this question is honest. An enchanted card is TWO physical cards: the
spell and the enchantment that upgrades it. Handing the enchant arm
extra castables for free would measure deck size, not enchanting. So
both arms get the same number of physical cards and differ only in how
those cards are spent — more copies of plain spells, or fewer spells
that stack higher.

Arms:
  plain        blades/traps/Feint, no enchants
  sharp        blades enchanted (stacking with their plain copies)
  potent       traps and Feint enchanted
  both         everything

READ THE SINGLE-ENCHANT ARMS WITH CARE. Slot-matching forces each arm
to cut something to pay for its enchants, and they do not all cut the
same thing — `sharp` funds blade enchants out of traps and Feint. So
`sharp` vs `potent` is NOT a clean read on blades vs traps; only
plain-vs-both is a single-variable comparison, and that is the one
quoted as the result.

Confidence: the +10% magnitude and the stacking rule are as reported by
the repo owner. MODELED: which ops the bonus lands on (see ENCHANTS).
ASSUMED AND UNPRICED: that enchanting costs deck slots but not TEMPO —
the enchanted card is treated as ready to cast on the turn it is drawn.
If applying an enchantment actually consumes the turn it is used on,
every number here is an upper bound. That assumption is flagged rather
than measured because the answer is a game-rules fact this repo does
not have a source for.
"""
import copy
import json
import time
from collections import Counter

from data_full import load_spells_full, LIVE_RULES
from data_full import load_bosses_full
from gear import loadout
from rulesets import CRIT_ERA
from w101_sim import (Sim, evaluate_paired, make_blade_stack,
                      register_enchants, enchanted_deck_size)

cards = dict(load_spells_full())
t0 = time.time()
POLS = {f"blade-stack({k})": make_blade_stack(k) for k in (1, 2, 3, 4)}
LVL = 100

register_enchants(cards, ["Fireblade", "Balanceblade"], "Sharpen Blade")
register_enchants(cards, ["Fire Trap", "Feint"], "Potent Trap")
register_enchants(cards, ["Fire Dragon", "Sunbird"], "Colossal")

# Every arm starts from the SAME buff package and the same nukes, and
# every arm pays for its enchants out of the SAME filler card. The
# first cut let each arm fund its enchants by cutting something
# different — `sharp` took it out of traps — which made the arms
# multi-variable and unreadable. Paying in filler makes each arm a
# single substitution against `plain`.
BASE = (["Fireblade"] * 3 + ["Balanceblade"] * 2 + ["Fire Trap"] * 3
        + ["Feint"] * 2 + ["Fire Dragon"] * 4 + ["Fire Elf"] * 2
        + ["Reshuffle"] * 2)
FILLER = "Sunbird"
SLOTS = 30

#  arm -> [(plain name, enchanted name, how many copies to swap)]
SETUPS = {
    "plain": [],
    "sharp": [("Fireblade", "Fireblade+sharp", 2),
              ("Balanceblade", "Balanceblade+sharp", 1)],
    "potent": [("Fire Trap", "Fire Trap+potent", 2),
               ("Feint", "Feint+potent", 1)],
    "colossal": [("Fire Dragon", "Fire Dragon+colossal", 2)],
    "all": [("Fireblade", "Fireblade+sharp", 2),
            ("Balanceblade", "Balanceblade+sharp", 1),
            ("Fire Trap", "Fire Trap+potent", 2),
            ("Feint", "Feint+potent", 1),
            ("Fire Dragon", "Fire Dragon+colossal", 2)],
}


def build(setup):
    dl = list(BASE)
    for plain, ench, n in SETUPS[setup]:
        for _ in range(n):
            dl[dl.index(plain)] = ench
    while enchanted_deck_size(dl) < SLOTS:
        dl.append(FILLER)
    while enchanted_deck_size(dl) > SLOTS and FILLER in dl:
        dl.remove(FILLER)
    return dl


DECKS = {k: build(k) for k in SETUPS}
print(f"every arm matched at {SLOTS} real deck slots; enchants are paid "
      f"for in {FILLER}s, so each arm is one substitution vs plain:",
      flush=True)
for k, dl in DECKS.items():
    print(f"  {k:<9} {len(dl):2d} castables / {enchanted_deck_size(dl)} "
          f"slots   {dl.count(FILLER)} {FILLER}s", flush=True)


def race(boss, deck, rules=CRIT_ERA, n=300, max_turns=80):
    lo = loadout(LVL, "fire", rules=rules)
    bb = copy.copy(boss)
    bb.dmg = 0
    sim = Sim(dict(cards), deck, "fire", bb, player_hp=10**9,
              rules=rules, player_stats=lo["player_stats"],
              power_pip=lo["power_pip"])
    r = evaluate_paired(sim, POLS, n=n, max_turns=max_turns)
    return max(r.items(), key=lambda kv: (round(kv[1]["win_rate"], 2),
                                          -kv[1]["mean_ttk"]))


# ---- does the stacking actually happen? -----------------------------
# Demonstrated directly rather than inferred from cast counts: put the
# plain blade and its sharpened copy on the same wizard and look at the
# charm list. This is the whole mechanic, so it is shown, not asserted.
lo = loadout(LVL, "fire", rules=CRIT_ERA)
demo = Sim(dict(cards), ["Fireblade", "Fireblade+sharp", "Fire Trap",
                         "Fire Trap+potent", "Sunbird"], "fire",
           copy.copy(load_bosses_full()[0]["Annoushka"]),
           player_hp=10**9, rules=CRIT_ERA,
           player_stats=lo["player_stats"])
st = demo.new_state()
st.player.charms.clear()
st.player.norm_pips = 10
for nm in ("Fireblade", "Fireblade+sharp"):
    demo.execute_ops(st, st.player, nm, cards[nm].source,
                     cards[nm].ops, "fire")
for nm in ("Fire Trap", "Fire Trap+potent"):
    demo.execute_ops(st, st.player, nm, cards[nm].source,
                     cards[nm].ops, "fire", target_idx=0)
print(f"\nstack check — charms on the wizard: "
      f"{[(h.name, h.percent) for h in st.player.charms]}", flush=True)
print(f"              wards on the boss:    "
      f"{[(h.name, h.percent) for h in st.enemies[0].wards]}", flush=True)
stacked = (len(st.player.charms) == 2 and len(st.enemies[0].wards) == 2)
print(f"    plain and enchanted coexist: {stacked} "
      f"(a second PLAIN copy would have been refused as a duplicate)",
      flush=True)

# ---- the headline: the boss that could not be killed ---------------
bosses, _ = load_bosses_full(rules=CRIT_ERA)
TARGETS = ["Annoushka", "Were-Bear Brute (Standard)", "High Priest Ixta"]
print("\nthe endgame bosses creature_stats.py called unreachable:",
      flush=True)
report = {}
for bn in TARGETS:
    b = bosses.get(bn)
    if b is None:
        continue
    row = {}
    for k, dl in DECKS.items():
        pol, st = race(b, dl)
        row[k] = {"win": st["win_rate"], "ttk": st["mean_ttk"],
                  "policy": pol}
    report[bn] = dict(row, hp=b.hp, resist=b.resist_map,
                      block=b.block_chance)
    cells = "   ".join(f"{k} {row[k]['win']*100:5.1f}%" for k in DECKS)
    print(f"  {bn:<28} hp {b.hp:6d}  {cells}", flush=True)

# ---- and on ordinary content, where TTK is defined -----------------
print("\nordinary content (TTK, lower is better):", flush=True)
ORDINARY = [b for b in bosses.values() if 6000 <= b.hp <= 11000]
ORDINARY.sort(key=lambda b: b.name)
ORDINARY = ORDINARY[:6]
ttk_rows = {}
for k, dl in DECKS.items():
    ttks = []
    for b in ORDINARY:
        _pol, st = race(b, dl, n=250)
        ttks.append(st["mean_ttk"])
    ok = [t for t in ttks if t == t]
    ttk_rows[k] = sum(ok) / len(ok) if ok else float("nan")
    print(f"  {k:<7} mean ttk {ttk_rows[k]:5.2f} over "
          f"{len(ok)}/{len(ORDINARY)} bosses", flush=True)
base = ttk_rows["plain"]
for k in DECKS:
    if k != "plain":
        print(f"    {k:<7} {base - ttk_rows[k]:+.2f} turns vs plain",
              flush=True)

json.dump({"setup": {"level": LVL, "slots": SLOTS,
                     "decks": DECKS,
                     "note": "every arm matched at the same real deck "
                             "slot count; enchants are paid for in "
                             "filler cards, so each arm is ONE "
                             "substitution against plain",
                     "confidence": "+10% and the stacking rule as "
                                   "reported by the repo owner; which "
                                   "ops the bonus lands on is MODELED; "
                                   "enchanting is assumed to cost deck "
                                   "slots but not tempo"},
           "endgame": report,
           "ordinary_mean_ttk": ttk_rows},
          open("results_enchants.json", "w", encoding="utf-8"), indent=1)
print(f"\nwrote results_enchants.json ({time.time()-t0:.0f}s)", flush=True)
