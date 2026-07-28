"""
How much validation does model selection here actually need?

The sequence-rung write-up had to be retracted twice, and the root
cause both times was the same: model selection on a small validation
set is a LOTTERY. Adversarial review quantified it — validation score
ranged 32.6-45.5 across checkpoints whose held-out score ranged
68-75.3, i.e. essentially uncorrelated — so whichever arm drew more
candidate checkpoints won, and the "winner" told you nothing about
the students.

This measures the apparatus instead of the students. One policy class
(linear BC), 11 checkpoints (epochs 1..11) — the exact arm whose
selection failed — scored under four validation budgets and against
the canonical 8-pair held-out benchmark. For each budget:

  spearman(val, test)  does validation RANK checkpoints at all?
  selection regret     held-out points lost by trusting val's pick,
                       vs the best checkpoint available
  oracle spread        best - worst checkpoint (what is at stake)

The output is a concrete rule for every future comparison in this
repo: the validation budget below which selection should not be used
at all (prefer a fixed schedule, and never compare arms with unequal
candidate pools).

The demonstration dataset is cached to offline_union.npz (gitignored,
regenerates from seeds) because building it costs ~280s.
"""
import json
import os
import random
import time

import numpy as np

from data_full import load_spells_full, LIVE_RULES
from deck_builder import legal_pool, sample_deck, random_boss, fine_tune
from deck_scorer import spearman
from generalist import GeneralistQ
from offline_rl import MAX_A, gen_dataset, gen_dataset_policy, train_bc
from search_policy import make_search_policy
from w101_sim import Sim, evaluate_paired, make_blade_stack

CACHE = "offline_union.npz"
cards = load_spells_full()
t0 = time.time()

if os.path.exists(CACHE):
    z = np.load(CACHE)
    union = {k: z[k] for k in z.files}
    print(f"[1/4] union loaded from {CACHE}: {len(union['G'])} "
          f"decisions", flush=True)
else:
    print("[1/4] building the union dataset (search + RL teachers)",
          flush=True)
    s_data, _ = gen_dataset_policy(
        cards, LIVE_RULES,
        make_search_policy(k=4, base=make_blade_stack(3)),
        n_pairs=16, eps_per_pair=100, seed=0)
    r_data, _ = gen_dataset(cards, LIVE_RULES, n_pairs=16,
                            eps_per_pair=150, seed=0)
    pad = s_data["phis"].shape[1] - MAX_A
    union = dict(
        phis=np.concatenate([s_data["phis"], np.pad(
            r_data["phis"], ((0, 0), (0, pad), (0, 0)))]),
        mask=np.concatenate([s_data["mask"], np.pad(
            r_data["mask"], ((0, 0), (0, pad)))]),
        chosen=np.concatenate([s_data["chosen"], r_data["chosen"]]),
        G=np.concatenate([s_data["G"], r_data["G"]]),
        won=np.concatenate([s_data["won"], r_data["won"]]),
        ep=np.concatenate([s_data["ep"],
                           r_data["ep"] + s_data["ep"].max() + 1]))
    np.savez_compressed(CACHE, **union)
    print(f"    {len(union['G'])} decisions, cached "
          f"({time.time()-t0:.0f}s)", flush=True)

print("[2/4] 11 candidate checkpoints (linear BC, epochs 1..11)",
      flush=True)
cands = []
for ne in range(1, 12):
    cands.append((ne, GeneralistQ(w=train_bc(union, winners_only=True,
                                             epochs=ne))))
print(f"    trained ({time.time()-t0:.0f}s)", flush=True)

print("[3/4] validation pools (seed 5150, feasibility-screened) and "
      "the canonical held-out benchmark (seed 777)", flush=True)
vrng = random.Random(5150)
val_pool, vtried = [], 0
schools = ("death", "myth", "balance", "fire", "ice", "storm", "life")
while len(val_pool) < 16 and vtried < 90:
    sc = schools[vtried % len(schools)]
    vtried += 1
    b = random_boss(vrng, f"seqval{vtried}")
    dl = sample_deck(legal_pool(cards, sc), sc, b, vrng)
    probe = Sim(dict(cards), dl, sc, b, player_hp=10**9,
                rules=LIVE_RULES)
    w = evaluate_paired(probe, {"h": make_blade_stack(3)},
                        n=150)["h"]["win_rate"]
    if 0.15 <= w <= 0.92:
        val_pool.append((sc, dl, b))
print(f"    {len(val_pool)} informative validation pairs", flush=True)

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
assert len(test_pairs) == 8
print(f"    8 held-out pairs ({time.time()-t0:.0f}s)", flush=True)


def score(model, pairs, n):
    tot = 0.0
    for sc, dl, b in pairs:
        sim = Sim(dict(cards), dl, sc, b, player_hp=10**9,
                  rules=LIVE_RULES)
        tot += evaluate_paired(sim, {"p": model.policy()},
                               n=n)["p"]["win_rate"]
    return tot / len(pairs)


print("[4/4] scoring every checkpoint on test, then under each "
      "validation budget", flush=True)
test = [score(m, test_pairs, 1500) for _, m in cands]
for (ne, _), t in zip(cands, test):
    print(f"    epochs {ne:>2}: held-out {t*100:5.1f}%", flush=True)

BUDGETS = [(4, 200), (8, 200), (8, 600), (16, 600)]
rows = []
for npairs, nf in BUDGETS:
    pool = val_pool[:npairs]
    val = [score(m, pool, nf) for _, m in cands]
    rho = spearman(val, test)
    pick = int(np.argmax(val))
    regret = max(test) - test[pick]
    rows.append(dict(pairs=npairs, fights_per_pair=nf,
                     total_fights=npairs * nf, spearman=rho,
                     picked_epochs=cands[pick][0],
                     picked_test=test[pick], best_test=max(test),
                     regret=regret, val=val))
    print(f"    budget {npairs:>2} pairs x {nf:>3} = "
          f"{npairs*nf:>5} fights: spearman {rho:+.2f}  picks "
          f"epochs={cands[pick][0]:>2} -> held-out "
          f"{test[pick]*100:5.1f}%  regret {regret*100:4.1f} pts",
          flush=True)

spread = max(test) - min(test)
print(f"\n    oracle spread across checkpoints: {spread*100:.1f} pts "
      f"(worst {min(test)*100:.1f}%, best {max(test)*100:.1f}%)",
      flush=True)
# MONOTONE rule: a budget counts as usable only if it AND every
# LARGER tested budget clear the bar. Without this the summary just
# reports whichever budget got lucky — the exact reasoning that made
# the sequence-rung claims collapse twice.
def _ok(r):
    return r["spearman"] >= 0.5 and r["regret"] <= 0.02


usable = None
for i, r in enumerate(rows):
    if all(_ok(x) for x in rows[i:]):
        usable = r
        break
lucky = [r for r in rows if _ok(r) and (usable is None or
                                        r["total_fights"] <
                                        usable["total_fights"])]
if usable is None:
    verdict = ("NO tested budget is reliably usable — prefer a fixed "
               "schedule, report the whole candidate curve instead of "
               "one selected point, and never compare arms with "
               "unequal candidate pools")
else:
    verdict = (f"selection is reliable from {usable['total_fights']} "
               f"validation fights (spearman "
               f"{usable['spearman']:+.2f}, regret "
               f"{usable['regret']*100:.1f} pts); smaller budgets are "
               f"not, even when one of them happens to score well")
if lucky:
    verdict += (" — note budgets " +
                ", ".join(str(r["total_fights"]) for r in lucky) +
                " clear the bar but LARGER ones do not, so their "
                "success is luck, not a floor")
print(f"    VERDICT: {verdict}", flush=True)

json.dump({"checkpoints": [ne for ne, _ in cands],
           "test": test, "oracle_spread": spread,
           "budgets": rows, "verdict": verdict},
          open("results_selection.json", "w", encoding="utf-8"),
          indent=1)
print(f"\nwrote results_selection.json ({time.time()-t0:.0f}s)",
      flush=True)
