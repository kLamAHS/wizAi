"""
Does student capacity actually bind? Linear vs recurrent, same data.

The offline ladder ended with every data source converging at ~75%
(BC from RL teachers 74.8, from search teachers 67.6, from their
union 73.1) against online RL's 76.7 and the scripted heuristic's
83.6 — the conclusion being that the binding constraint had moved
from teacher quality to STUDENT CAPACITY. That is a claim about the
policy class, and this file tests it the only honest way: hold the
data, the features, and the benchmark fixed, and change ONLY the
student.

  linear BC-filtered   — the incumbent, retrained here on the union
                         so the comparison is same-data
  ablation (U = 0)     — the SAME episode-level Adam/BPTT optimizer
                         with the recurrence pinned off: still exactly
                         a linear policy
  recurrent BC-filtered — RecurrentQ (w_eff = w0 + U h), warm-started
                         from the linear solution

The ablation is the whole point: w0 trains in both arms, so without
it a recurrent-vs-linear gap cannot be attributed to memory.

RESULT (results_sequence_model.json), budget-matched at 11 candidate
checkpoints per arm and selected at the 9,600-fight budget
selection_study.py certifies: linear 75.1 / ablation 75.7 /
recurrent 71.5 on the held-out 8; 54.8 / 76.8 / 76.7 on the X-pip
bar. Capacity is NOT the constraint (a memoryless linear policy
clears the bar the class supposedly could not express) and recurrence
buys nothing: −4.2 in aggregate, dead level on the sequencing bar.
Two earlier versions of this file claimed otherwise — a 7.6-point
"optimizer" gap (artifact of 5 vs 11 candidates) and a +4.1 recurrent
edge on the bar (artifact of an 800-fight selection budget that
selection_study.py later showed to be luck).

Evaluation reproduces the exact 8 held-out pairs of results_offline
(seed 777, per-pair expert >= 20%), plus the frozen X-pip bar. Both
students get the same validation-based model selection on SEPARATE
pairs (seed 5150), with the same number of candidates; the test pairs
are never used for selection, and the held-out table is scored with
paired seeds so the arms face identical draws.
"""
import json
import os
import random
import time

import numpy as np

from data_full import load_spells_full, LIVE_RULES
from deck_builder import legal_pool, sample_deck, random_boss, fine_tune
from generalist import GeneralistQ
from offline_rl import MAX_A, gen_dataset, gen_dataset_policy, train_bc
from search_policy import make_search_policy
from seq_policy import RecurrentQ, episodes_from, train_seq_bc
from w101_sim import Sim, evaluate, evaluate_paired, make_blade_stack

cards = load_spells_full()
t0 = time.time()

CACHE = "offline_union.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE)
    union = {k: z[k] for k in z.files}
    print(f"[1/5] union loaded from {CACHE}: {len(union['G'])} "
          f"decisions", flush=True)
else:
    print("[1/5] demonstrations: search teachers + RL teachers, "
          "unioned", flush=True)
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
n_ep = len(episodes_from(union, winners_only=True))
print(f"    {len(union['G'])} decisions, {n_ep} winning episodes "
      f"({time.time()-t0:.0f}s)", flush=True)

print("[2/5] validation pairs for model selection (seed 5150 — "
      "disjoint from the test pairs)", flush=True)
# screened to be INFORMATIVE (an unwinnable or saturated pair
# cannot rank two students) and sized per selection_study.py: 16
# pairs x 600 fights is the smallest budget whose val->test rank
# correlation held up at every LARGER budget too. The earlier 4x200
# budget scored well once, but 8x200 and 8x600 both cost 5.5 points,
# so that success was luck rather than a floor.
vrng = random.Random(5150)
val_pairs, vtried = [], 0
while len(val_pairs) < 16 and vtried < 90:
    sc = ("death", "myth", "balance", "fire", "ice",
          "storm", "life")[vtried % 7]
    vtried += 1
    b = random_boss(vrng, f"seqval{vtried}")
    dl = sample_deck(legal_pool(cards, sc), sc, b, vrng)
    probe = Sim(dict(cards), dl, sc, b, player_hp=10**9,
                rules=LIVE_RULES)
    w = evaluate_paired(probe, {"h": make_blade_stack(3)},
                        n=150)["h"]["win_rate"]
    if 0.15 <= w <= 0.92:
        val_pairs.append((sc, dl, b))
        print(f"    val pair {sc:<8} {b.name:<18} probe "
              f"{w*100:5.1f}%", flush=True)


def val_score(model):
    tot = 0.0
    for sc, dl, b in val_pairs:
        sim = Sim(dict(cards), dl, sc, b, player_hp=10**9,
                  rules=LIVE_RULES)
        st = evaluate_paired(sim, {"p": model.policy()}, n=600)["p"]
        tot += st["win_rate"]
    return tot / len(val_pairs)


print("[3/5] incumbent: linear BC-filtered on the union, with the "
      "SAME validation-based model selection the recurrent student "
      "gets (equal treatment, else the comparison is rigged)",
      flush=True)
# BUDGET-MATCHED: the recurrent/ablation arms get 11 candidates
# (warm start + 10 epochs), so the linear arm gets 11 too. With an
# unequal grid (5 vs 11) the extra draws were worth +7 points to
# whichever arm had more, and validation here is nearly uncorrelated
# with test — selection is a lottery whose odds are set by the pool.
best_lin = (-1.0, None, None)
lin_curve = {}
for ne in range(1, 12):
    w_try = train_bc(union, winners_only=True, epochs=ne)
    sc_try = val_score(GeneralistQ(w=w_try))
    lin_curve[ne] = sc_try
    print(f"    linear epochs={ne:>2}: val {sc_try*100:5.1f}%",
          flush=True)
    if sc_try > best_lin[0]:
        best_lin = (sc_try, w_try, ne)
w_lin, linear = best_lin[1], GeneralistQ(w=best_lin[1])
print(f"    linear selected: epochs={best_lin[2]}, val "
      f"{best_lin[0]*100:.1f}%", flush=True)

print("[4/5] recurrent BC-filtered, warm-started from the linear "
      "solution (BPTT over whole episodes)", flush=True)
t1 = time.time()
seq = train_seq_bc(union, w_linear=w_lin, H=8, epochs=10, lr=0.01,
                   batch=16, seed=0, log=print, val=val_score)
seq.save("policy_seq.json")
print(f"    trained ({time.time()-t1:.0f}s)", flush=True)

print("[4b/5] ABLATION — same optimizer, U frozen at 0 (exactly "
      "linear). w0 trains in both arms, so this is what separates "
      "'recurrence helped' from 'the episode-level optimizer helped'",
      flush=True)
abl = train_seq_bc(union, w_linear=w_lin, H=8, epochs=10, lr=0.01,
                   batch=16, seed=0, log=print, val=val_score,
                   freeze_U=True)
abl.save("policy_seq_ablation.json")

print("[5/5] the same 8 held-out pairs as results_offline (seed 777):",
      flush=True)
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
    st = evaluate_paired(sim, {"linear": linear.policy(),
                               "ablation": abl.policy(),
                               "recurrent": seq.policy()}, n=1500)
    row = dict(boss=boss.name, school=sc,
               linear=[st["linear"]["win_rate"],
                       st["linear"]["mean_ttk"]],
               ablation=[st["ablation"]["win_rate"],
                         st["ablation"]["mean_ttk"]],
               recurrent=[st["recurrent"]["win_rate"],
                          st["recurrent"]["mean_ttk"]],
               old_bc_filtered=prev["BC-filtered"][0],
               heuristic=prev["heuristic"][0])
    rows.append(row)
    print(f"    {sc:<8} vs {boss.name:<18} "
          f"linear {row['linear'][0]*100:5.1f}%  "
          f"ablation {row['ablation'][0]*100:5.1f}%  "
          f"recurrent {row['recurrent'][0]*100:5.1f}%  "
          f"[heuristic {row['heuristic']*100:5.1f}%]", flush=True)

assert len(rows) == 8, f"only {len(rows)} held-out pairs collected"
means = {k: sum(r[k][0] if isinstance(r[k], list) else r[k]
                for r in rows) / len(rows)
         for k in ("linear", "ablation", "recurrent",
                   "old_bc_filtered", "heuristic")}
print("    means (fresh, paired seeds): linear "
      f"{means['linear']*100:.1f}%  ablation(U=0) "
      f"{means['ablation']*100:.1f}%  recurrent "
      f"{means['recurrent']*100:.1f}%", flush=True)
print("    means (historical, unpaired, different run — context "
      f"only): old BC-filtered {means['old_bc_filtered']*100:.1f}%  "
      f"heuristic {means['heuristic']*100:.1f}%", flush=True)

print("\n[bonus] frozen X-pip bar. Reference numbers come from "
      "DIFFERENT runs and protocols — per-deck RL 85.1% and online "
      "generalist 53.4% are unpaired n=2000, search(k=16) 72.0% is "
      "paired n=300; the rows below are paired n=1500. Treat the "
      "references as context, not as a like-for-like column:",
      flush=True)
xrng = random.Random(4242)
xpair = None
for i, sc in enumerate(("fire", "storm", "death", "ice", "life",
                        "balance")):
    b = random_boss(xrng, f"held{i}")
    dl = sample_deck(legal_pool(cards, sc), sc, b, xrng)
    if i == 5:
        xpair = (sc, dl, b)
sc, dl, xboss = xpair
assert xboss.name == "held5-ice-3100" and sc == "balance", \
    f"frozen X-pip pair drifted: {sc}/{xboss.name}"
xsim = Sim(dict(cards), dl, sc, xboss, player_hp=10**9, rules=LIVE_RULES)
xst = evaluate_paired(xsim, {"linear": linear.policy(),
                             "ablation": abl.policy(),
                             "recurrent": seq.policy()}, n=1500)
xbar = {k: [v["win_rate"], v["mean_ttk"]] for k, v in xst.items()}
for k, v in xbar.items():
    print(f"    {k:<10} win {v[0]*100:5.1f}%", flush=True)


def _no_nan(o):
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_no_nan(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


json.dump(_no_nan({"n_decisions": int(len(union["G"])),
                   "n_win_episodes": n_ep, "held_out": rows,
                   "means": means,
                   "selection": {"linear_val_curve": lin_curve,
                                 "linear_selected_epochs": best_lin[2],
                                 "candidates_per_arm": 11,
                                 "val_pairs": [b.name for _, _, b
                                               in val_pairs]},
                   "xpip_bar": {"pair": f"{sc}/{xboss.name}",
                                "deck": sorted(set(dl)), "n": 1500,
                                "protocol": "paired seeds",
                                "arms": xbar,
                                "reference_other_runs": {
                                    "per_deck_RL_unpaired_n2000": 0.851,
                                    "online_generalist_unpaired_n2000":
                                        0.534,
                                    "search_k16_paired_n300": 0.72}}}),
          open("results_sequence_model.json", "w", encoding="utf-8"),
          indent=1, allow_nan=False)
print(f"\nwrote results_sequence_model.json, policy_seq.json "
      f"(total {time.time()-t0:.0f}s)", flush=True)
