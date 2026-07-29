"""
These fights are not 1v1.

`bosses_clean.json` names the creatures that fight alongside each boss
for 421 of them, and the loader was dropping the field. Every result in
this repo that used a real boss was therefore fighting it alone.

Two kinds of companion, and the split matters:

  RESOLVED (25% of references) — the name matches another creature in
    the file, so it arrives with real scraped stats. These turn out to
    be PEERS: same rank as the boss, ~0.95x its health. Genuine
    multi-boss fights, not underlings.
  UNRESOLVED (75%) — a generic mob with no page of its own. The fight
    is still not 1v1, so `encounter(..., synth=True)` builds a stand-in
    at MINION_HP_SHARE of the boss's health, a couple of ranks below.
    That share is the most arbitrary number in the loader, so it gets a
    sensitivity sweep here rather than a footnote.

Measured: what the missing companions were worth, and whether the deck
builder adapts to them on its own.
"""
import copy
import json
import random
import statistics as st
import time

from data_full import (LIVE_RULES, MINION_HP_SHARE, encounter,
                       load_bosses_full, load_spells_full)
from deck_builder import build_deck
from gear import loadout
from w101_sim import Sim, evaluate_paired, make_blade_stack, make_survival

cards = load_spells_full()
t0 = time.time()
POLS = {f"blade-stack({k})": make_blade_stack(k) for k in (1, 2, 3)}
POLS.update({f"triage({k})": make_survival(make_blade_stack(k))
             for k in (1, 2)})
DECK = ["Fireblade", "Fireblade", "Fire Trap", "Feint", "Tower Shield",
        "Tower Shield", "Fire Dragon", "Fire Dragon", "Sunbird",
        "Sunbird", "Pixie", "Fire Elf"]
LVL = 100
lo = loadout(LVL, "fire", rules=LIVE_RULES)
bosses, registry = load_bosses_full()
report = {}


def fight(boss, companions, n=300):
    sim = Sim(dict(cards), DECK, "fire", copy.copy(boss),
              player_hp=lo["player_hp"], rules=LIVE_RULES,
              player_stats=lo["player_stats"],
              power_pip=lo["power_pip"],
              enemies=[copy.copy(c) for c in companions])
    r = evaluate_paired(sim, POLS, n=n)
    return max(r.values(), key=lambda v: (round(v["win_rate"], 2),
                                          -v["mean_ttk"]))


# ---- coverage -------------------------------------------------------
withm = [b for b in bosses.values() if b.minions]
refs = [m for b in withm for m in b.minions]
resolved = [m for m in refs if m in bosses]
print(f"[coverage] {len(withm)} of {len(bosses)} creatures fight with "
      f"company ({len(withm)/len(bosses)*100:.0f}%)", flush=True)
print(f"           {len(refs)} companion references, "
      f"{len(resolved)} resolve to real stats "
      f"({len(resolved)/len(refs)*100:.0f}%)", flush=True)
sizes = [len(b.minions) for b in withm]
print(f"           party size beyond the boss: median {st.median(sizes):.0f}, "
      f"max {max(sizes)}", flush=True)
report["coverage"] = {"bosses_with_company": len(withm),
                      "bosses_total": len(bosses),
                      "references": len(refs),
                      "resolved": len(resolved),
                      "median_party": st.median(sizes)}

# ---- what was the missing company worth? ----------------------------
# Measured two ways, because win rate alone is useless here: with a
# fixed flat-damage boss the fights are CLIFF-like (a scan of the HP
# bands returns 100% below ~9k and 0% above ~14k, with a thin mixed
# strip between), so a win-rate-only probe reports noise plus the
# occasional total flip. The race isolates the extra WORK the company
# adds; the survival arm, screened to contested fights only, isolates
# the extra DANGER.
def race(boss, companions, n=250):
    sim = Sim(dict(cards), DECK, "fire", copy.copy(boss),
              player_hp=10**9, rules=LIVE_RULES,
              player_stats=lo["player_stats"],
              power_pip=lo["power_pip"],
              enemies=[copy.copy(c) for c in companions])
    r = evaluate_paired(sim, POLS, n=n, max_turns=80)
    return max(r.values(), key=lambda v: (round(v["win_rate"], 2),
                                          -v["mean_ttk"]))


print(f"\n[Q1a] extra WORK — same boss, immortal wizard, TTK to clear "
      f"the whole encounter\n", flush=True)
cands = [b for b in withm if 3000 <= b.hp <= 12000 and b.rank]
cands.sort(key=lambda b: b.name)
random.Random(7).shuffle(cands)
real_rows, synth_rows = [], []
for b in cands:
    if len(real_rows) >= 5 and len(synth_rows) >= 5:
        break
    n_real = sum(1 for m in b.minions if m in bosses)
    bucket = real_rows if n_real else synth_rows
    if len(bucket) >= 5:
        continue
    solo = race(b, [])
    _b, comps = encounter(b.name, bosses)
    full = race(b, comps)
    if solo["mean_ttk"] != solo["mean_ttk"] or \
            full["mean_ttk"] != full["mean_ttk"]:
        continue                      # unclearable either way
    bucket.append({"boss": b.name, "hp": b.hp, "party": len(comps),
                   "resolved": n_real, "solo_ttk": solo["mean_ttk"],
                   "party_ttk": full["mean_ttk"],
                   "extra_turns": full["mean_ttk"] - solo["mean_ttk"]})
for label, rows in (("real peers", real_rows),
                    ("inferred mobs", synth_rows)):
    print(f"  companions that {label}:", flush=True)
    for r in rows:
        print(f"    {r['boss']:<28} +{r['party']} "
              f"({r['resolved']} real)   ttk {r['solo_ttk']:5.2f} -> "
              f"{r['party_ttk']:5.2f}   ({r['extra_turns']:+5.2f})",
              flush=True)
    if rows:
        m = st.median(r["extra_turns"] for r in rows)
        print(f"    median extra turns: {m:+.2f}\n", flush=True)
report["q1a_race"] = {"real_peers": real_rows, "inferred_mobs": synth_rows}

print(f"[Q1b] extra DANGER — mortal wizard, only fights that are "
      f"CONTESTED solo (30-95%)\n", flush=True)
rows, drops, scanned = [], [], 0
for b in cands:
    if len(rows) >= 8 or scanned > 60:
        break
    scanned += 1
    solo = fight(b, [], n=250)
    if not 0.30 <= solo["win_rate"] <= 0.95:
        continue
    _b, comps = encounter(b.name, bosses)
    full = fight(b, comps, n=250)
    d = (solo["win_rate"] - full["win_rate"]) * 100
    rows.append({"boss": b.name, "hp": b.hp, "party": len(comps),
                 "resolved": sum(1 for m in b.minions if m in bosses),
                 "solo": solo["win_rate"],
                 "with_company": full["win_rate"], "drop": d})
    drops.append(d)
    print(f"    {b.name:<28} +{len(comps)}   "
          f"{solo['win_rate']*100:5.1f}% -> "
          f"{full['win_rate']*100:5.1f}%   ({-d:+6.1f} pts)", flush=True)
if drops:
    print(f"\n    {len(rows)} contested of {scanned} scanned; mean "
          f"win-rate drop {sum(drops)/len(drops):+.1f} points "
          f"(median {st.median(drops):+.1f})", flush=True)
else:
    print("    no contested encounter found in the scan", flush=True)
report["q1b_survival"] = {"rows": rows, "scanned": scanned,
                          "mean_drop": (sum(drops) / len(drops)
                                        if drops else None)}

# ---- how much of that rides on the modeled HP share? ----------------
print(f"\n[Q2] sensitivity to MINION_HP_SHARE (modeled at "
      f"{MINION_HP_SHARE}); measured on the RACE, which is continuous, "
      f"and only over bosses whose companions are inferred\n",
      flush=True)
sens = {}
sample = [r["boss"] for r in synth_rows]
for share in (0.0, 0.2, 0.35, 0.5, 0.7):
    ttks = []
    for name in sample:
        b, comps = encounter(name, bosses, hp_share=share,
                             synth=share > 0)
        t = race(b, comps, n=200)["mean_ttk"]
        if t == t:
            ttks.append(t)
    sens[share] = sum(ttks) / len(ttks) if ttks else float("nan")
    label = "no companions" if share == 0 else f"share {share}"
    print(f"    {label:<16} mean ttk {sens[share]:5.2f}", flush=True)
report["q2_hp_share_sensitivity"] = sens

# ---- does the builder adapt? ---------------------------------------
print(f"\n[Q3] given the real party, does the DECK change?\n", flush=True)
target = max((b for b in withm
              if 4000 <= b.hp <= 7000 and len(b.minions) >= 2),
             key=lambda b: b.hp)
_b, comps = encounter(target.name, bosses)
print(f"    {target.name}: {target.hp} HP + {len(comps)} companions",
      flush=True)
from deck_builder import legal_pool                       # noqa: E402
pool = {**cards, **legal_pool(cards, "fire", level=LVL, enchants=True)}
q3 = {}
for arm, enemies in (("solo", None), ("real party", comps)):
    dl, w, m, _ = build_deck(
        cards, "fire", copy.copy(target), LIVE_RULES, n_candidates=50,
        top_k=3, capacity=20, seed=23, level=LVL, enchants=True,
        enemies=[copy.copy(c) for c in enemies] if enemies else None,
        player_hp=lo["player_hp"], player_stats=lo["player_stats"],
        power_pip=lo["power_pip"], log=None)
    dmg = [pool[n] for n in dl if pool[n].kind in ("damage", "drain")]
    aoe = sum(1 for c in dmg if c.aoe)
    q3[arm] = {"deck": dl, "win": w, "ttk": m, "cards": len(dl),
               "damage_cards": len(dmg), "aoe_cards": aoe}
    print(f"    built for {arm:<11} {len(dl):2d} cards, "
          f"{len(dmg)} damage ({aoe} AoE)   win {w*100:5.1f}%  "
          f"ttk {m:5.2f}", flush=True)
print(f"\n    NOTE: fire's best nukes are already AoE, so 'AoE share of "
      f"damage cards' saturates at 100% in both arms and cannot answer "
      f"this. What moves is the COUNT of damage cards carried.",
      flush=True)
report["q3_builder"] = dict(q3, boss=target.name, party=len(comps))

json.dump(report, open("results_minions.json", "w", encoding="utf-8"),
          indent=1)
print(f"\nwrote results_minions.json ({time.time()-t0:.0f}s)", flush=True)
