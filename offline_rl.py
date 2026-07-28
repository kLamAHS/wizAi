"""
Offline RL ladder (roadmap 6): expert data -> filtered behavior
cloning -> conservative offline RL, benchmarked on held-out pairs.

The design isolates one question: with the POLICY CLASS held fixed
(the generalist's linear card-vs-state features), is it better to
learn from per-deck EXPERT demonstrations (each a tabular QAgent
fine-tuned on its own (deck, boss) pair — they know draw-specific
lines the linear class can't represent) or from online RL's own
exploration (train_generalist)? Everything evaluates zero-shot on
held-out (deck, boss) pairs none of the learners saw.

Rungs implemented, all numpy, no deep-learning dependency:
  BC-all      — behavior cloning on every logged decision (softmax
                over the legal actions' features, max likelihood of
                the expert's choice).
  BC-filtered — same, cloning only WINNING episodes: losers teach
                losing habits, so filter first (the cheapest big win
                in offline RL).
  CQL-lite    — conservative fitted-Q: regress w.phi(chosen) onto the
                backward-MC return while pushing down the logsumexp
                of Q over all legal alternatives (linear-class CQL
                flavor; named -lite honestly).

Data notes: behavior policy is the expert with eps=0.1 exploration
(pure-greedy data has no coverage and offline RL degenerates on it);
episodes log the feature matrix of every legal action, the choice,
and the return-to-go. The dataset regenerates deterministically from
seeds and is NOT committed (see .gitignore); results_offline.json is.
Sequence models remain roadmap.
"""
import json
import random

import numpy as np

from w101_sim import Sim, evaluate, make_blade_stack
from rl_agent import PASS, MAX_TURNS, FAIL_PENALTY, apply_action
from generalist import phi, FEATS, GeneralistQ, G_SCALE
from deck_builder import legal_pool, sample_deck, random_boss, fine_tune

MAX_A = 10          # pad/truncate the per-step legal-action axis


# ------------------------------------------------------------- data gen

def _legal_pairs(sim, s):
    """(action_object, name) for PASS + castable distinct cards."""
    out, seen = [(PASS, PASS)], set()
    for c in s.hand:
        if c.name not in seen and sim.can_cast(s, c):
            seen.add(c.name)
            out.append((c, c.name))
    return out[:MAX_A]


def rollout(sim, agent, seed, eps=0.1):
    """One logged episode under the expert + exploration noise.
    Returns (steps, won): steps = [(phi_matrix, mask, chosen_idx)]."""
    sim.rng = random.Random(seed)
    rng = random.Random(seed + 1)
    s = sim.new_state()
    steps, won = [], False
    while True:
        acts = _legal_pairs(sim, s)
        P = np.zeros((MAX_A, len(FEATS)), np.float32)
        mask = np.zeros(MAX_A, bool)
        for i, (a, _) in enumerate(acts):
            P[i] = phi(sim, s, a)
            mask[i] = True
        names = [n for _, n in acts]
        if rng.random() < eps:
            pick = rng.randrange(len(acts))
        else:
            k = agent.feat.key(sim, s)
            best = max(names, key=lambda n: agent.Q[(k, n)])
            pick = names.index(best)
        steps.append((P, mask, pick))
        apply_action(sim, s, names[pick])
        if s.boss_hp <= 0:
            won = True
            break
        sim.end_round(s)
        if s.player_hp <= 0 or s.turn >= MAX_TURNS:
            break
    return steps, won


def gen_dataset_policy(cards, rules, teacher, n_pairs=16,
                       eps_per_pair=100, seed=0, log=None, eps=0.02,
                       max_a=16, feas_win=0.10):
    """Search/scripted-teacher variant of gen_dataset: the behavior
    policy is any (sim, s) -> card | (card, tgt) | None policy. Noise
    is eps=0.02 per the brittleness lesson (10% noise on a tight line
    destroys the wins it demonstrates). Pairs are feasibility-screened
    with a cheap scripted probe so the data isn't despair."""
    from generalist import legal_actions, match_action, phi
    from w101_sim import evaluate, make_blade_stack

    rng = random.Random(seed)
    schools = ("fire", "ice", "storm", "myth", "life", "death", "balance")
    pools = {sc: legal_pool(cards, sc) for sc in schools}
    P, M, C, G, W = [], [], [], [], []
    pairs, made = [], 0
    while made < n_pairs:
        sc = schools[len(pairs) % len(schools)]
        boss = random_boss(rng, f"sdata{len(pairs)}")
        dl = sample_deck(pools[sc], sc, boss, rng)
        pairs.append((sc, dl, boss))
        sim = Sim(dict(cards), dl, sc, boss, player_hp=10**9,
                  rules=rules)
        probe, _ = evaluate(sim, make_blade_stack(3), n=200)
        if probe < feas_win:
            if log:
                log(f"    skip {boss.name} ({sc}): probe "
                    f"{probe*100:.0f}% — no signal")
            continue
        made += 1
        t_wins = 0
        for e in range(eps_per_pair):
            sim.rng = random.Random(seed * 10**6 + made * 10**4 + e)
            s = sim.new_state()
            steps, won = [], False
            while True:
                acts = legal_actions(sim, s)[:max_a]
                pick = match_action(acts, teacher(sim, s))
                if rng.random() < eps:
                    pick = rng.randrange(len(acts))
                Pm = np.zeros((max_a, len(FEATS)), np.float32)
                mask = np.zeros(max_a, bool)
                for i, act in enumerate(acts):
                    Pm[i] = phi(sim, s, act)
                    mask[i] = True
                steps.append((Pm, mask, pick))
                taken = acts[pick]
                if taken == PASS:
                    pass
                elif isinstance(taken, tuple):
                    sim.cast(s, taken[0], target=taken[1])
                else:
                    sim.cast(s, taken)
                if not any(en.alive for en in s.enemies):
                    won = True
                    break
                sim.end_round(s)
                if s.player_hp <= 0 or s.turn >= MAX_TURNS:
                    break
            t_wins += won
            g = 0.0 if won else -FAIL_PENALTY
            for Pm, mask, pick in reversed(steps):
                g = -1.0 + g
                P.append(Pm)
                M.append(mask)
                C.append(pick)
                G.append(g / 10.0)
                W.append(won)
        if log:
            log(f"    pair {made:>2} {boss.name:<18} {sc:<8} teacher "
                f"{t_wins/eps_per_pair*100:5.1f}%  "
                f"({len(P)} steps total)")
    return dict(phis=np.stack(P), mask=np.stack(M),
                chosen=np.array(C, np.int16),
                G=np.array(G, np.float32), won=np.array(W, bool)), pairs


def gen_dataset(cards, rules, n_pairs=16, eps_per_pair=150, seed=0,
                log=None):
    """Per pair: fine-tune a tabular expert, then log noisy-expert
    rollouts. Pairs where the expert can't win at all are kept out of
    the dataset (they contain no signal, only despair)."""
    rng = random.Random(seed)
    schools = ("fire", "ice", "storm", "myth", "life", "death", "balance")
    pools = {sc: legal_pool(cards, sc) for sc in schools}
    P, M, C, G, W = [], [], [], [], []
    pairs = []
    made = 0
    while made < n_pairs:
        sc = schools[len(pairs) % len(schools)]
        boss = random_boss(rng, f"data{len(pairs)}")
        dl = sample_deck(pools[sc], sc, boss, rng)
        pairs.append((sc, dl, boss))
        w, m, agent = fine_tune(cards, dl, sc, boss, rules,
                                seed=seed + made)
        if w < 0.05:
            if log:
                log(f"    skip {boss.name} ({sc}): expert win "
                    f"{w*100:.0f}% — no signal")
            continue
        made += 1
        sim = Sim(dict(cards), dl, sc, boss, player_hp=10**9,
                  rules=rules)
        for e in range(eps_per_pair):
            steps, won = rollout(sim, agent, seed * 10**6 + made * 10**4
                                 + e)
            g = 0.0 if won else -FAIL_PENALTY
            for Pm, mk, ch in reversed(steps):
                g = -1.0 + g
                P.append(Pm)
                M.append(mk)
                C.append(ch)
                G.append(g / G_SCALE)
                W.append(won)
        if log:
            log(f"    pair {made:>2} {boss.name:<18} {sc:<8} expert "
                f"{w*100:5.1f}%/{m:5.2f}  ({len(P)} steps total)")
    return dict(phis=np.stack(P), mask=np.stack(M),
                chosen=np.array(C, np.int16),
                G=np.array(G, np.float32), won=np.array(W, bool)), pairs


# ------------------------------------------------------------- learners

def _iterate(data, grad_step, epochs, lr0, batch, seed, keep=None):
    idx = np.arange(len(data["G"]))
    if keep is not None:
        idx = idx[keep]
    rng = np.random.default_rng(seed)
    w = np.zeros(len(FEATS))
    for ep in range(epochs):
        lr = lr0 * (1 - 0.9 * ep / epochs)
        rng.shuffle(idx)
        for lo in range(0, len(idx), batch):
            b = idx[lo:lo + batch]
            w -= lr * grad_step(w, b)
    return w


def train_bc(data, winners_only=False, epochs=8, lr0=0.5, batch=256,
             seed=0):
    """Softmax behavior cloning over legal actions."""
    P, M, C = data["phis"], data["mask"], data["chosen"]

    def grad(w, b):
        logits = P[b] @ w
        logits[~M[b]] = -1e9
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        pr = z / z.sum(axis=1, keepdims=True)
        pr[np.arange(len(b)), C[b]] -= 1.0
        return (pr[:, :, None] * P[b]).sum(axis=(0, 1)) / len(b)

    keep = data["won"] if winners_only else None
    return _iterate(data, grad, epochs, lr0, batch, seed, keep)


def train_cql(data, alpha=1.0, epochs=8, lr0=0.1, batch=256, seed=0):
    """Conservative fitted-Q: MSE to the return on the LOGGED action,
    plus alpha * (logsumexp Q over legal - Q(logged)) pushing down
    whatever the data can't vouch for."""
    P, M, C, G = data["phis"], data["mask"], data["chosen"], data["G"]

    def grad(w, b):
        q = P[b] @ w
        q[~M[b]] = -1e9
        rows = np.arange(len(b))
        qc = q[rows, C[b]]
        phic = P[b][rows, C[b]]
        g_mse = 2 * (qc - G[b])[:, None] * phic
        z = np.exp(q - q.max(axis=1, keepdims=True))
        pr = z / z.sum(axis=1, keepdims=True)
        g_cons = (pr[:, :, None] * P[b]).sum(axis=1) - phic
        return (g_mse + alpha * g_cons).sum(axis=0) / len(b)

    return _iterate(data, grad, epochs, lr0, batch, seed)


# ------------------------------------------------------------- benchmark

if __name__ == "__main__":
    import time

    from data_full import load_spells_full, LIVE_RULES
    from generalist import train_generalist

    cards = load_spells_full()
    t0 = time.time()
    print("[1/4] experts + logged rollouts: 16 pairs x 150 episodes "
          "(eps=0.1 behavior noise)", flush=True)
    data, pairs = gen_dataset(cards, LIVE_RULES, n_pairs=16,
                              eps_per_pair=150, seed=0, log=print)
    np.savez_compressed("offline_data.npz", **data)
    print(f"    {len(data['G'])} decisions logged "
          f"({data['won'].mean()*100:.0f}% from wins)  "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("[2/4] training BC-all, BC-filtered, CQL-lite "
          "(same linear class as the generalist)", flush=True)
    pols = {
        "BC-all": GeneralistQ(w=train_bc(data)),
        "BC-filtered": GeneralistQ(w=train_bc(data, winners_only=True)),
        "CQL-lite": GeneralistQ(w=train_cql(data, alpha=1.0)),
    }
    print("[3/4] online generalist baseline (40k episodes, its own "
          "exploration)", flush=True)
    pols["online-RL"] = train_generalist(cards, episodes=40000, seed=0,
                                         rules=LIVE_RULES)

    print("[4/4] zero-shot evaluation on 8 HELD-OUT pairs — feasibility-"
          "filtered (a per-pair expert must clear 20%, else the pair "
          "is despair, not signal); the expert column is the trained-"
          "on-that-pair upper bound, not zero-shot:", flush=True)
    hrng = random.Random(777)
    schools = ("fire", "storm", "death", "ice", "life", "myth",
               "balance")
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
        heur = max((make_blade_stack(k) for k in (1, 2, 3)),
                   key=lambda p: evaluate(sim, p, n=300)[0])
        row = dict(boss=boss.name, school=sc, expert=[we, me],
                   heuristic=list(evaluate(sim, heur, n=1500)))
        for name, pol in pols.items():
            row[name] = list(evaluate(sim, pol.policy(), n=1500))
        rows.append(row)
        cells = "  ".join(f"{n} {row[n][0]*100:5.1f}%"
                          for n in ("heuristic", *pols, "expert"))
        print(f"    {sc:<8} vs {boss.name:<18} {cells}", flush=True)
    means = {n: sum(r[n][0] for r in rows) / len(rows)
             for n in ("heuristic", *pols, "expert")}
    print("    means: " + "  ".join(f"{n} {v*100:.1f}%"
                                    for n, v in means.items()),
          flush=True)

    json.dump({"n_decisions": int(len(data["G"])),
               "pct_from_wins": float(data["won"].mean()),
               "held_out": rows, "means": means},
              open("results_offline.json", "w", encoding="utf-8"),
              indent=1)
    for name, pol in pols.items():
        if name != "online-RL":
            pol.save(f"policy_{name.lower().replace('-', '_')}.json")
    print("\nwrote results_offline.json, offline_data.npz, "
          "policy_*.json", flush=True)
