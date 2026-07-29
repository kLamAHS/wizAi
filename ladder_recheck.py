"""
Which offline-ladder conclusions survive the corrected methodology?

The original ladder (results_offline.json) reported BC-all 48.0,
BC-filtered 74.8, CQL-lite 71.3, and drew two conclusions: filtering
demonstrations is the biggest lever (+27), and conservatism (CQL)
trails filtered BC by ~3.5. Both were measured the old way — ONE
fixed schedule per learner (8 epochs, no selection), scored unpaired.

selection_study.py then showed the noise floor: checkpoint choice
alone moves held-out score by 6.9 points. A 27-point gap clears that
comfortably; a 3.5-point gap does not. So this re-runs all three
learners under the corrected regime:

  - the same RL-teacher dataset the original ladder used (cached)
  - 11 candidate checkpoints EACH (epochs 1..11), budget-matched
  - selection at the certified 9,600-fight validation budget
  - the full candidate curve reported (min/median/max on test), so
    the within-learner spread sits next to the between-learner gap
  - paired seeds on the canonical 8 held-out pairs

A conclusion survives only if the gap between learners exceeds the
spread WITHIN them. Anything else was a lottery result.
"""
import json
import os
import random
import statistics
import time

import numpy as np

from data_full import load_spells_full, LIVE_RULES
from deck_builder import legal_pool, sample_deck, random_boss, fine_tune
from generalist import GeneralistQ
from offline_rl import gen_dataset, train_bc, train_cql
from w101_sim import Sim, evaluate_paired, make_blade_stack

CACHE = "offline_rl_teacher.npz"
cards = load_spells_full()
t0 = time.time()

if os.path.exists(CACHE):
    z = np.load(CACHE)
    data = {k: z[k] for k in z.files}
    print(f"[1/5] RL-teacher dataset from {CACHE}: "
          f"{len(data['G'])} decisions", flush=True)
else:
    print("[1/5] regenerating the original ladder's RL-teacher data",
          flush=True)
    data, _ = gen_dataset(cards, LIVE_RULES, n_pairs=16,
                          eps_per_pair=150, seed=0)
    np.savez_compressed(CACHE, **data)
    print(f"    {len(data['G'])} decisions, cached "
          f"({time.time()-t0:.0f}s)", flush=True)

print("[2/5] 11 candidates per learner (epochs 1..11)", flush=True)
LEARNERS = {
    "BC-all": lambda ne: train_bc(data, winners_only=False, epochs=ne),
    "BC-filtered": lambda ne: train_bc(data, winners_only=True,
                                       epochs=ne),
    "CQL-lite": lambda ne: train_cql(data, alpha=1.0, epochs=ne),
}
cands = {name: [(ne, GeneralistQ(w=fn(ne))) for ne in range(1, 12)]
         for name, fn in LEARNERS.items()}
print(f"    {sum(len(v) for v in cands.values())} models trained "
      f"({time.time()-t0:.0f}s)", flush=True)

print("[3/5] validation pool (16 pairs, certified budget) and the "
      "canonical held-out 8", flush=True)
vrng = random.Random(5150)
schools = ("death", "myth", "balance", "fire", "ice", "storm", "life")
val_pairs, vtried = [], 0
while len(val_pairs) < 16 and vtried < 90:
    sc = schools[vtried % len(schools)]
    vtried += 1
    b = random_boss(vrng, f"seqval{vtried}")
    dl = sample_deck(legal_pool(cards, sc), sc, b, vrng)
    probe = Sim(dict(cards), dl, sc, b, player_hp=10**9,
                rules=LIVE_RULES)
    w = evaluate_paired(probe, {"h": make_blade_stack(3)},
                        n=150)["h"]["win_rate"]
    if 0.15 <= w <= 0.92:
        val_pairs.append((sc, dl, b))

hrng = random.Random(777)
tschools = ("fire", "storm", "death", "ice", "life", "myth", "balance")
test_pairs, tried = [], 0
while len(test_pairs) < 8 and tried < 40:
    sc = tschools[tried % len(tschools)]
    tried += 1
    boss = random_boss(hrng, f"held{tried}")
    dl = sample_deck(legal_pool(cards, sc), sc, boss, hrng)
    we, _, _ = fine_tune(cards, dl, sc, boss, LIVE_RULES,
                         seed=100 + tried)
    if we >= 0.2:
        test_pairs.append((sc, dl, boss))
assert len(test_pairs) == 8 and len(val_pairs) == 16
print(f"    ready ({time.time()-t0:.0f}s)", flush=True)


def score(model, pairs, n):
    tot = 0.0
    for sc, dl, b in pairs:
        sim = Sim(dict(cards), dl, sc, b, player_hp=10**9,
                  rules=LIVE_RULES)
        tot += evaluate_paired(sim, {"p": model.policy()},
                               n=n)["p"]["win_rate"]
    return tot / len(pairs)


print("[4/5] candidate curves on test (n=500) + validated selection "
      "(16 x 600), then the selected model at n=1500", flush=True)
old = json.load(open("results_offline.json", encoding="utf-8"))
report = {}
for name, lst in cands.items():
    curve = [score(m, test_pairs, 500) for _, m in lst]
    val = [score(m, val_pairs, 600) for _, m in lst]
    pick = int(np.argmax(val))
    final = score(lst[pick][1], test_pairs, 1500)
    report[name] = dict(
        curve=curve, val=val, picked_epochs=lst[pick][0],
        selected_test=final, curve_min=min(curve), curve_max=max(curve),
        curve_median=statistics.median(curve),
        spread=max(curve) - min(curve),
        original=old["means"].get(name))
    r = report[name]
    print(f"    {name:<12} curve {r['curve_min']*100:5.1f}-"
          f"{r['curve_max']*100:5.1f}% (median "
          f"{r['curve_median']*100:5.1f}, spread "
          f"{r['spread']*100:4.1f})  selected epochs="
          f"{r['picked_epochs']:>2} -> {final*100:5.1f}%  "
          f"[original {r['original']*100:5.1f}%]", flush=True)

print("[5/5] which conclusions survive?", flush=True)
verdicts = []
pairs_to_check = [("BC-filtered", "BC-all",
                   "filtering demonstrations is the biggest lever"),
                  ("BC-filtered", "CQL-lite",
                   "filtered BC beats conservative fitted-Q")]
for a, b, claim in pairs_to_check:
    gap = report[a]["selected_test"] - report[b]["selected_test"]
    noise = max(report[a]["spread"], report[b]["spread"])
    ok = gap > noise
    verdicts.append(dict(claim=claim, gap=gap, noise_floor=noise,
                         survives=bool(ok)))
    print(f"    {'SURVIVES' if ok else 'NOT SUPPORTED'}: {claim}\n"
          f"        gap {gap*100:+.1f} pts vs within-learner spread "
          f"{noise*100:.1f} pts", flush=True)

json.dump({"learners": report, "verdicts": verdicts,
           "protocol": {"candidates_per_learner": 11,
                        "selection_fights": 16 * 600,
                        "curve_n": 500, "final_n": 1500,
                        "paired": True}},
          open("results_ladder_recheck.json", "w", encoding="utf-8"),
          indent=1)
print(f"\nwrote results_ladder_recheck.json ({time.time()-t0:.0f}s)",
      flush=True)
