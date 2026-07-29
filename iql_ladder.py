"""
Does per-decision credit beat per-episode credit? (roadmap 6, rung 4)

The offline ladder's existing rungs assign credit bluntly. BC-filtered
keeps or drops a decision by whether its EPISODE was won — every move
in a lucky win is taught, every move in an unlucky loss is thrown
away. CQL-lite is conservative about actions nobody played, which
means evaluating Q off-distribution.

IQL offers a third scheme: score each decision by its own ADVANTAGE
over a state baseline, never querying an unplayed action. The baseline
is fit by EXPECTILE regression, which is what makes it optimistic
rather than average — above tau = 0.5 the upside residuals carry more
weight, so V drifts from "what the behavior policy averages here"
toward "what a good action from here gets".

That gives a clean two-step ablation, and the second step is the one
worth running:

  BC-filtered   -> AWR (tau=0.5)   does PER-DECISION credit beat
                                   per-episode credit at all?
  AWR (tau=0.5) -> IQL (tau>0.5)   does the EXPECTILE add anything
                                   over a plain mean baseline?

METHODOLOGY, which this repo has had to learn the hard way. Three
earlier conclusions here were published and then retracted after
adversarial review showed the effects sat under the noise floor. So:
every learner is trained at several seeds, all learners see the same
held-out pairs on paired seeds, the within-learner spread across seeds
is reported next to the between-learner gaps, and NO arm is selected
on the held-out set — every arm that was run is printed, including the
ones that lose. A gap is only called real if it clears the pooled
within-learner spread.

beta is fixed from the TRAINING data's advantage scale (1/sd), not
tuned against held-out performance, so the arms stay honest.
"""
import json
import os
import random
import time

import numpy as np

from data_full import load_spells_full, LIVE_RULES
from deck_builder import legal_pool, sample_deck, random_boss, fine_tune
from generalist import FEATS, GeneralistQ
from offline_rl import (gen_dataset, train_bc, train_cql, train_iql,
                        train_expectile_v, advantages, state_phi)
from w101_sim import Sim, evaluate, make_blade_stack

CACHE = "offline_data.npz"
SEEDS = (0, 1, 2, 3, 4)
# Sixteen held-out pairs, split in half. The first eight SELECT (which
# arm, which beta); the last eight REPORT. This repo has retracted
# three conclusions that turned out to be selection artifacts, and the
# first cut of this very probe reproduced the mistake in miniature:
# beta was fixed a priori at 0.57, lost by 8 points, and a sweep then
# found beta=2 winning by 6.8 — a number chosen on the same pairs it
# was scored on, which is not a result. Hence the split.
N_HELD = 16
N_VAL = 8
cards = load_spells_full()
t0 = time.time()


def load_or_build():
    """The cache is keyed on the feature width: the featurizer grew
    when school pips landed, and a stale 26-wide dataset silently
    mismatches a 28-wide learner."""
    if os.path.exists(CACHE):
        d = dict(np.load(CACHE))
        if d["phis"].shape[-1] == len(FEATS):
            print(f"  reusing {CACHE} ({len(d['G'])} decisions)",
                  flush=True)
            return d
        print(f"  {CACHE} is {d['phis'].shape[-1]}-wide but FEATS is "
              f"{len(FEATS)} — regenerating", flush=True)
    d, _ = gen_dataset(cards, LIVE_RULES, n_pairs=16, eps_per_pair=150,
                       seed=0, log=print)
    np.savez_compressed(CACHE, **d)
    return d


print("[1/4] dataset", flush=True)
data = load_or_build()
print(f"    {len(data['G'])} decisions "
      f"({data['won'].mean()*100:.0f}% from wins)  "
      f"({time.time()-t0:.0f}s)", flush=True)

# beta from the training data's own advantage scale — never from
# held-out performance.
S = state_phi(data)
adv_sd = float(advantages(data, train_expectile_v(data, tau=0.7, S=S),
                          S).std())
BETA = round(1.0 / max(adv_sd, 1e-6), 2)
print(f"[2/4] advantage sd {adv_sd:.3f} -> beta {BETA} (fixed a priori)",
      flush=True)

# Diagnostic, printed BEFORE any policy is evaluated, because it
# predicts the result rather than explaining it after the fact: how
# much of the return does the state baseline actually explain, and how
# much of the advantage is just the win/loss label in disguise?
G = data["G"]
diag = {}
for tau in (0.5, 0.7, 0.9):
    v = train_expectile_v(data, tau=tau, S=S)
    a = advantages(data, v, S)
    r2 = 1 - ((G - S @ v) ** 2).sum() / ((G - G.mean()) ** 2).sum()
    diag[tau] = {"r2": float(r2),
                 "adv_wins": float(a[data["won"]].mean()),
                 "adv_losses": float(a[~data["won"]].mean())}
    print(f"    tau {tau}: V explains R^2 {r2:+.3f} of the return; "
          f"mean advantage {a[data['won']].mean():+.2f} on wins vs "
          f"{a[~data['won']].mean():+.2f} on losses", flush=True)
print(f"    raw return: {G[data['won']].mean():+.2f} on wins vs "
      f"{G[~data['won']].mean():+.2f} on losses (sd {G.std():.2f}) — "
      f"the outcome, not the move, is where the signal lives",
      flush=True)

ARMS = {
    "BC-all": lambda s: train_bc(data, seed=s),
    "BC-filtered": lambda s: train_bc(data, winners_only=True, seed=s),
    "CQL-lite": lambda s: train_cql(data, alpha=1.0, seed=s),
    "AWR (tau=.5)": lambda s: train_iql(data, tau=0.5, beta=BETA, seed=s),
    "IQL (tau=.7)": lambda s: train_iql(data, tau=0.7, beta=BETA, seed=s),
    "IQL (tau=.9)": lambda s: train_iql(data, tau=0.9, beta=BETA, seed=s),
    # The first run of this probe stopped at the six arms above and
    # found IQL LOSING to BC-filtered by 8.2 points. Two confounds had
    # to be ruled out before that could be published as a fact about
    # credit assignment rather than about this configuration:
    #   beta — fixed a priori at 0.57, which is a gentle reweighting;
    #          a sharper one might separate good decisions properly.
    #          The whole sweep is reported, never the best of it.
    #   composition — AWR trains on ALL episodes while BC-filtered
    #          drops losers outright. The soft weighting may simply be
    #          a weaker filter than the hard label, in which case the
    #          two should COMPOSE rather than compete.
    "IQL b=2": lambda s: train_iql(data, tau=0.7, beta=2.0, seed=s),
    "IQL b=5": lambda s: train_iql(data, tau=0.7, beta=5.0, seed=s),
    "IQL b=15": lambda s: train_iql(data, tau=0.7, beta=15.0, seed=s),
    "IQL+filter": lambda s: train_iql(data, tau=0.7, beta=BETA, seed=s,
                                      winners_only=True),
    "IQL+filter b=5": lambda s: train_iql(data, tau=0.7, beta=5.0,
                                          seed=s, winners_only=True),
}

print(f"[3/4] held-out pairs (expert must clear 20%, else the pair is "
      f"despair not signal)", flush=True)
# The selection is fully deterministic from seed 777 — random_boss and
# sample_deck are cheap replays of one RNG stream, and the only costly
# step is the per-pair expert. So cache the experts' win rates by try
# index; the pairs themselves are re-derived, never stored.
EXPERT_CACHE = "iql_expert_cache.json"
_ecache = {}
if os.path.exists(EXPERT_CACHE):
    _ecache = json.load(open(EXPERT_CACHE, encoding="utf-8"))

hrng = random.Random(777)
schools = ("fire", "storm", "death", "ice", "life", "myth", "balance")
held, tried = [], 0
while len(held) < N_HELD and tried < 70:
    sc = schools[tried % len(schools)]
    tried += 1
    boss = random_boss(hrng, f"held{tried}")
    dl = sample_deck(legal_pool(cards, sc), sc, boss, hrng)
    if str(tried) in _ecache:
        we = _ecache[str(tried)]
    else:
        we, _me, _ = fine_tune(cards, dl, sc, boss, LIVE_RULES,
                               seed=100 + tried)
        _ecache[str(tried)] = we
        json.dump(_ecache, open(EXPERT_CACHE, "w", encoding="utf-8"))
    if we < 0.2:
        continue
    held.append({"school": sc, "deck": dl, "boss": boss,
                 "expert": we})
    print(f"    {sc:<8} vs {boss.name:<18} expert {we*100:5.1f}%",
          flush=True)

VAL, TEST = held[:N_VAL], held[N_VAL:]
print(f"\n[4/4] every arm at {len(SEEDS)} seeds, zero-shot on "
      f"{len(VAL)} SELECT pairs and {len(TEST)} REPORT pairs\n",
      flush=True)


def score(pol, pairs):
    wins = []
    for h in pairs:
        sim = Sim(dict(cards), h["deck"], h["school"], h["boss"],
                  player_hp=10**9, rules=LIVE_RULES)
        wins.append(evaluate(sim, pol, n=1500)[0])
    return sum(wins) / len(wins)


report = {}
for name, fit in ARMS.items():
    v_seed, t_seed = [], []
    for s in SEEDS:
        pol = GeneralistQ(w=fit(s)).policy()
        v_seed.append(score(pol, VAL))
        t_seed.append(score(pol, TEST))
    report[name] = {
        "val": sum(v_seed) / len(v_seed),
        "test": sum(t_seed) / len(t_seed),
        "val_half_spread": (max(v_seed) - min(v_seed)) / 2,
        "test_half_spread": (max(t_seed) - min(t_seed)) / 2,
        "val_per_seed": v_seed, "test_per_seed": t_seed}
    r = report[name]
    print(f"  {name:<15} select {r['val']*100:5.1f}%  "
          f"report {r['test']*100:5.1f}%  "
          f"+/-{r['test_half_spread']*100:4.1f} across seeds", flush=True)

floor = max(v["test_half_spread"] for v in report.values())
print(f"\n  pooled within-learner noise floor: "
      f"+/-{floor*100:.1f} points on the report split", flush=True)

# The honest headline: pick on SELECT, quote on REPORT. The oracle row
# is printed only to price the selection, never to claim.
picked = max(report, key=lambda k: report[k]["val"])
oracle = max(report, key=lambda k: report[k]["test"])
base = "BC-filtered"
gap = (report[picked]["test"] - report[base]["test"]) * 100
print(f"\n  selected on the SELECT split: {picked}", flush=True)
print(f"  its REPORT score: {report[picked]['test']*100:.1f}%  vs "
      f"{base} {report[base]['test']*100:.1f}%   ({gap:+.1f} pts, "
      f"{'clears' if abs(gap) > floor*100 else 'UNDER'} the floor)",
      flush=True)
print(f"  best-on-REPORT would have been {oracle} at "
      f"{report[oracle]['test']*100:.1f}% — the "
      f"{(report[oracle]['test']-report[picked]['test'])*100:.1f}-point "
      f"difference is the price of honest selection", flush=True)

COMPARISONS = [("BC-filtered", "AWR (tau=.5)",
                "per-decision credit vs per-episode"),
               ("AWR (tau=.5)", "IQL (tau=.7)",
                "expectile vs mean baseline"),
               ("IQL (tau=.7)", "IQL (tau=.9)",
                "more optimistic expectile"),
               ("IQL (tau=.7)", "IQL b=5",
                "was the a-priori beta the problem?"),
               ("IQL b=5", "IQL+filter b=5",
                "do the two credit schemes compose?"),
               ("BC-filtered", "IQL b=2",
                "advantage weighting vs the episode filter")]
print("", flush=True)
verdicts = {}
for a, b, why in COMPARISONS:
    g = (report[b]["test"] - report[a]["test"]) * 100
    real = abs(g) > floor * 100
    verdicts[f"{a} -> {b}"] = {"gap_points": g,
                               "clears_noise_floor": bool(real),
                               "question": why}
    print(f"  {a:<15} -> {b:<15} {g:+6.1f} pts   "
          f"{'REAL' if real else 'under the noise floor'}   ({why})",
          flush=True)

json.dump({"setup": {"seeds": list(SEEDS), "beta": BETA,
                     "advantage_sd": adv_sd,
                     "n_decisions": int(len(data["G"])),
                     "pct_from_wins": float(data["won"].mean()),
                     "select_pairs": [
                         {"school": h["school"], "boss": h["boss"].name,
                          "expert": h["expert"]} for h in VAL],
                     "report_pairs": [
                         {"school": h["school"], "boss": h["boss"].name,
                          "expert": h["expert"]} for h in TEST],
                     "note": "arms are selected on the SELECT split and "
                             "quoted on the REPORT split; every arm run "
                             "is listed either way"},
           "baseline_diagnostic": {str(k): v for k, v in diag.items()},
           "return_by_outcome": {
               "wins": float(G[data["won"]].mean()),
               "losses": float(G[~data["won"]].mean()),
               "sd": float(G.std())},
           "arms": report,
           "noise_floor_points": floor * 100,
           "selected_on_val": picked,
           "best_on_test_oracle": oracle,
           "selection_price_points": (report[oracle]["test"]
                                      - report[picked]["test"]) * 100,
           "comparisons": verdicts},
          open("results_iql.json", "w", encoding="utf-8"), indent=1)
print(f"\nwrote results_iql.json ({time.time()-t0:.0f}s)", flush=True)
