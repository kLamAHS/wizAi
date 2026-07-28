"""
Offline ladder, stronger teachers: search-generated expert data.

The first offline run's honest ceiling was its teachers — 8k-episode
tabular experts averaging BELOW the scripted heuristic, so their
clones could only inherit mediocrity (garbage-ceiling, not
garbage-in). Decision-time search now exceeds scripted play, so the
original roadmap clause ("search-generated expert data") is finally
actionable: same clones, same held-out pairs, better demonstrations.

Changes from the first run, each a recorded lesson applied:
  - teacher: search(k=4, blade-stack base) instead of per-pair RL
  - behavior noise eps=0.02, not 0.1 (10% noise on a tight line
    destroys the wins it exists to demonstrate)
  - data pairs feasibility-screened with a cheap scripted probe

Evaluation reproduces the EXACT held-out pairs of results_offline
(seed 777, per-pair expert >= 20% filter), so the columns are
directly comparable:
  old table: BC-all 48.0 | BC-filt 74.8 | CQL 71.3 | online 76.7 |
             experts 78.9 | heuristic 83.6
"""
import json
import random
import time

from data_full import load_spells_full, LIVE_RULES
from deck_builder import legal_pool, sample_deck, random_boss, fine_tune
from generalist import GeneralistQ
from offline_rl import gen_dataset_policy, train_bc, train_cql
from search_policy import make_search_policy
from w101_sim import Sim, evaluate, make_blade_stack

cards = load_spells_full()
teacher = make_search_policy(k=4, base=make_blade_stack(3))

t0 = time.time()
print("[1/3] search-teacher demonstrations: 16 feasible pairs x 100 "
      "episodes, eps=0.02", flush=True)
data, pairs = gen_dataset_policy(cards, LIVE_RULES, teacher,
                                 n_pairs=16, eps_per_pair=100, seed=0,
                                 log=print)
print(f"    {len(data['G'])} decisions "
      f"({data['won'].mean()*100:.0f}% from wins)  "
      f"({time.time()-t0:.0f}s)", flush=True)

print("[2/4] training BC-filtered and CQL-lite on the search data",
      flush=True)
pols = {
    "BC-filt(search)": GeneralistQ(w=train_bc(data, winners_only=True)),
    "CQL-lite(search)": GeneralistQ(w=train_cql(data, alpha=1.0)),
}

print("[3/4] union cell: regenerate the RL-teacher dataset and clone "
      "the COMBINED data — separates coverage (union should win) "
      "from representability (union stays flat)", flush=True)
import numpy as np
from offline_rl import gen_dataset, MAX_A

rl_data, _ = gen_dataset(cards, LIVE_RULES, n_pairs=16,
                         eps_per_pair=150, seed=0)
pad = data["phis"].shape[1] - MAX_A
union = dict(
    phis=np.concatenate([data["phis"], np.pad(
        rl_data["phis"], ((0, 0), (0, pad), (0, 0)))]),
    mask=np.concatenate([data["mask"], np.pad(
        rl_data["mask"], ((0, 0), (0, pad)))]),
    chosen=np.concatenate([data["chosen"], rl_data["chosen"]]),
    G=np.concatenate([data["G"], rl_data["G"]]),
    won=np.concatenate([data["won"], rl_data["won"]]))
print(f"    union: {len(union['G'])} decisions "
      f"({union['won'].mean()*100:.0f}% from wins)", flush=True)
pols["BC-filt(union)"] = GeneralistQ(
    w=train_bc(union, winners_only=True))

print("[4/4] zero-shot on the SAME 8 held-out pairs as "
      "results_offline (rebuilt from seed 777):", flush=True)
old = json.load(open("results_offline.json", encoding="utf-8"))
hrng = random.Random(777)
schools = ("fire", "storm", "death", "ice", "life", "myth", "balance")
rows, tried = [], 0
while len(rows) < 8 and tried < 40:
    sc = schools[tried % len(schools)]
    tried += 1
    boss = random_boss(hrng, f"held{tried}")
    dl = sample_deck(legal_pool(cards, sc), sc, boss, hrng)
    we, me, _ = fine_tune(cards, dl, sc, boss, LIVE_RULES,
                          seed=100 + tried)
    if we < 0.2:
        continue
    sim = Sim(dict(cards), dl, sc, boss, player_hp=10**9,
              rules=LIVE_RULES)
    prev = next(r for r in old["held_out"] if r["boss"] == boss.name)
    row = dict(boss=boss.name, school=sc,
               old_bc_filtered=prev["BC-filtered"][0],
               old_online=prev["online-RL"][0])
    for name, pol in pols.items():
        row[name] = list(evaluate(sim, pol.policy(), n=1500))
    rows.append(row)
    print(f"    {sc:<8} vs {boss.name:<18} "
          f"BC-filt(search) {row['BC-filt(search)'][0]*100:5.1f}%  "
          f"CQL(search) {row['CQL-lite(search)'][0]*100:5.1f}%  "
          f"union {row['BC-filt(union)'][0]*100:5.1f}%  "
          f"[old BC-filt {row['old_bc_filtered']*100:5.1f}%  "
          f"online {row['old_online']*100:5.1f}%]", flush=True)

means = {}
for key in ("BC-filt(search)", "CQL-lite(search)", "BC-filt(union)"):
    means[key] = sum(r[key][0] for r in rows) / len(rows)
means["old BC-filtered"] = sum(r["old_bc_filtered"] for r in rows) \
    / len(rows)
means["old online-RL"] = sum(r["old_online"] for r in rows) / len(rows)
print("    means: " + "  ".join(f"{k} {v*100:.1f}%"
                                for k, v in means.items()), flush=True)
print("    (old table context: heuristic 83.6%, per-pair experts "
      "78.9%)", flush=True)

json.dump({"n_decisions": int(len(data["G"])),
           "pct_from_wins": float(data["won"].mean()),
           "held_out": rows, "means": means},
          open("results_offline_search.json", "w", encoding="utf-8"),
          indent=1)
for name, pol in pols.items():
    pol.save(f"policy_{name.split('(')[0].lower().replace('-', '_')}"
             "_search.json")
print("\nwrote results_offline_search.json, policy_*_search.json",
      flush=True)
