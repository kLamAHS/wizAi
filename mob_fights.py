"""
Mob fights, opening move (roadmap 4): target switching.

The living-boss experiment ended on a cliff: one 300-HP healer
acolyte (which genuinely heals its boss) dropped a ~33%-winnable solo
fight to 0%, because every policy in the repo aims at enemy 0. The
engine was never the obstacle — casts take a target index and wards
hang per enemy — the POLICIES were target-blind.

with_focus() adds the report's team-fight rule as a wrapper: support
enemies (healer/buffer/debuffer archetypes) die first, then the
lowest-HP attacker. Arms, level-20 base-stat death wizard, living
Krokopatra, paired seeds n=2000:

  boss alone            — the solo baseline for reference
  +healer, target-blind — the 0% cliff, reproduced
  +healer, focus        — kill-the-healer-first: does the fight
                          come back?
  +2 healers, focus     — the nightmare check: does the rule scale
                          or does double sustain outpace focus?

Pilots are scripted (triage and blade-stack under the focus wrapper);
the learned pilots stay target-blind for now — giving the tabular
and linear agents a target dimension is the next rung, and the
scripted result here sets the bar they must clear.
"""
import copy
import json

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from player_curves import school_hp, base_power_pip
from w101_sim import (Sim, Boss, evaluate_paired, make_blade_stack,
                      make_survival, with_focus)

cards = load_spells_full()
bosses, _ = load_bosses_full()

living = copy.copy(bosses["Krokopatra"])
living.dmg = 0
living.pool = ["Thunder Snake", "Lightning Bats", "Storm Shark",
               "Stormblade", "Storm Shield"]
living.archetype = "hitter"
living.discipline = 0.7


def acolyte(n):
    a = Boss(f"krok-acolyte-{n}", 300, "life", 0,
             pool=["Sprite", "Pixie"], archetype="healer")
    a.resist_map = {}
    a.boost_map = {}
    return a


LEVEL = 20
HP, PP = school_hp("death", LEVEL), base_power_pip(LEVEL)
solo_deck = json.load(open("results_living.json", encoding="utf-8")
                      )["levels"]["20"]["living"]["deck"]
print(f"level-{LEVEL} death wizard ({HP} HP, {PP*100:.0f}% base pips)",
      flush=True)
print(f"solo-built deck (size {len(solo_deck)}): "
      f"{sorted(set(solo_deck))}", flush=True)

# the solo deck traced out of ammunition at boss=388 with the healer
# dead: the size penalty that makes solo decks elegant under-equips a
# 1,260-HP encounter. Build FOR the encounter: focus-wrapped screen
# proxies (a target-blind screen ranks noise here), top_k=1 because
# the stage-2 learned pilots are still target-blind — the screen is
# the only aligned judge available.
from deck_builder import build_deck

print("\nbuilding a deck FOR the boss+healer encounter:", flush=True)
mob_deck, _, _, _ = build_deck(cards, "death", living, LIVE_RULES,
                               n_candidates=60, top_k=1, seed=21,
                               level=LEVEL, player_hp=HP, power_pip=PP,
                               capacity=14, enemies=[acolyte(1)],
                               log=lambda m: print("  ", m, flush=True))
print(f"mob-built deck (size {len(mob_deck)}): "
      f"{sorted(set(mob_deck))}", flush=True)

# the builder's screen collapses on the mortal encounter (every
# candidate 0% -> the size tiebreak favors SMALL decks, exactly wrong
# for a two-enemy HP pool). The hand-built magazine isolates the
# ammunition variable: same spells, maximum copies.
ammo_deck = (["Banshee"] * 3 + ["Ghoul"] * 3 + ["Sacrifice"] * 3 +
             ["Dark Sprite"] * 2 +
             ["Deathblade", "Balanceblade", "Curse"])

triage = make_survival(make_blade_stack(3))
ARMS = [
    ("boss alone", solo_deck, [],
     {"triage": triage}),
    ("+healer, blind", solo_deck, [acolyte(1)],
     {"triage": triage}),
    ("+healer, solo deck", solo_deck, [acolyte(1)],
     {"focus(triage)": with_focus(triage)}),
    ("+healer, mob deck", mob_deck, [acolyte(1)],
     {"focus(triage)": with_focus(triage),
      "focus(bs3)": with_focus(make_blade_stack(3))}),
    ("+healer, ammo deck", ammo_deck, [acolyte(1)],
     {"focus(bs2)": with_focus(make_blade_stack(2)),
      "blind(bs2)": make_blade_stack(2)}),
    ("+2 healers, ammo deck", ammo_deck, [acolyte(1), acolyte(2)],
     {"focus(bs2)": with_focus(make_blade_stack(2))}),
]
report = {}
for name, deck, extra, pilots in ARMS:
    report[name] = {"deck_size": len(deck)}
    # mortal = the era-stat feasibility verdict; immortal = the pure
    # tempo view that isolates the targeting mechanism from the
    # you-die-during-the-detour confound
    for view, php in (("mortal", HP), ("immortal", 10**9)):
        sim = Sim(dict(cards), deck, "death", living, player_hp=php,
                  power_pip=PP, rules=LIVE_RULES, enemies=extra)
        st = evaluate_paired(sim, pilots, n=2000)
        report[name][view] = {p: {k: s[k] for k in
                                  ("win_rate", "mean_ttk", "p90_ttk")}
                              for p, s in st.items()}
        for p, s in st.items():
            mean = s["mean_ttk"]
            print(f"   {name:<18} {view:<8} [{p:<13}] win "
                  f"{s['win_rate']*100:5.1f}%"
                  + (f"  mean {mean:5.2f}" if mean == mean else ""),
                  flush=True)


def _no_nan(o):
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_no_nan(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


json.dump(_no_nan({"level": LEVEL, "player_hp": HP,
                   "solo_deck": solo_deck, "mob_deck": mob_deck,
                   "ammo_deck": ammo_deck,
                   "arms": report}),
          open("results_mob.json", "w", encoding="utf-8"), indent=1,
          allow_nan=False)
print("\nwrote results_mob.json")
