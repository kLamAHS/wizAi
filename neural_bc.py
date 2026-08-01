"""Behavior-cloning the search into an entity net -- the torch side.

The first neural seat is the rollout continuation (see
deimos_bridge/neural_net.py for why). This script produces the weights
that module runs: it generates decisions from the search on randomized
boards, trains a small MLP to reproduce them, and exports numpy-ready
JSON. Torch is needed HERE and nowhere else -- the repo's tests, the
GUI and the live loop stay dependency-free.

    python3 neural_bc.py gen  N SEED OUT.npz     # torch-free
    python3 neural_bc.py train OUT.json IN.npz [IN2.npz ...]

TEACHER: greedy_ttk -- the policy the live loop actually drives with --
at horizon 6 (three-quarters of decisions; the measured buff-spam
lever) and 12 (the rest, for patience examples), default continuation.
Cloning the OUTER argmin of the search gives the student the search's
judgement at a thousandth of its cost; whether that judgement survives
distillation is measured in-sim afterwards, not assumed from held-out
accuracy.

BOARDS: randomized across all seven wizard schools, levels 1-60 legal
pools, sampled decks, 1-3 enemies of random school and health, incoming
damage 0-150 per mob, player health and gear varied. Saturated and
hopeless boards are kept: the teacher still chooses, and "which move"
is the label, not "did it win".
"""
import random
import sys

import numpy as np

sys.path.insert(0, ".")

SCHOOLS = ("fire", "ice", "storm", "myth", "life", "death", "balance")


def _teacher_rows(sim, teacher, cap=12, driver=None):
    """[(feature matrix, chosen index)] for one fight.

    `driver=None` is plain behaviour cloning: the teacher plays and its
    states are recorded. Passing a student policy is the DAgger round:
    the STUDENT drives -- so the states are the ones the student
    actually reaches, drift included -- and the teacher only labels
    them. Round-one BC measured exactly the failure DAgger exists for:
    competitive on easy boards, collapsed on hard ones (7.0 vs 17.5 in
    the continuation seat), which is compounding error off the
    teacher's state distribution.
    """
    from deimos_bridge.neural_net import decision_matrix
    from deimos_bridge.policies import _split

    rows = []

    def recording(sim_, s):
        choice = teacher(sim_, s)
        if len(rows) < cap:
            cands, X = decision_matrix(sim_, s)
            if choice is None:
                rows.append((X, len(cands) - 1))
            else:
                card, t = _split(choice)
                idx = next((i for i, (c, tt) in enumerate(cands)
                            if c is not None and c.name == card.name
                            and tt == t), None)
                if idx is not None:
                    rows.append((X, idx))
        return choice if driver is None else driver(sim_, s)

    sim.run(recording, max_turns=14)
    return rows


def generate(n_decisions, seed, out_path, student_weights=None):
    from data_full import LIVE_RULES, load_spells_full
    from deck_builder import legal_pool, sample_deck
    from deimos_bridge.policies import greedy_ttk
    from w101_sim import Boss, Sim

    student = None
    if student_weights:
        from deimos_bridge.neural_net import net_policy
        student = net_policy(student_weights)

    cards = load_spells_full()
    rng = random.Random(seed)
    Xs, ys, lens = [], [], []
    got = 0
    while got < n_decisions:
        school = rng.choice(SCHOOLS)
        pool = legal_pool(cards, school, level=rng.randint(1, 60))
        hp = rng.randint(120, 2400)
        dmg = rng.choice((0, 30, 60, 90, 120, 150))
        boss = Boss(name="t", hp=hp, school=rng.choice(SCHOOLS), dmg=dmg)
        extra = [Boss(name=f"t{i}", hp=int(hp * rng.uniform(0.35, 1.0)),
                      school=rng.choice(SCHOOLS), dmg=dmg)
                 for i in range(1, rng.randint(1, 3))]
        deck = sample_deck(pool, school, boss, rng,
                           rng.choice((12, 16, 20)), 3, lethal=dmg > 0)
        if not deck:
            continue
        teacher = greedy_ttk(6 if rng.random() < 0.75 else 12)
        sim = Sim(cards, deck, school, boss, enemies=extra,
                  player_hp=rng.randint(450, 2600),
                  player_stats={"damage": {"*": rng.uniform(0, 0.45)},
                                "accuracy": rng.uniform(0, 0.10)},
                  rules=LIVE_RULES,
                  rng=random.Random(rng.randrange(10 ** 9)))
        try:
            fight = _teacher_rows(sim, teacher, driver=student)
        except Exception:
            continue          # one degenerate board must not stop a shard
        for X, y in fight:
            Xs.append(X.astype(np.float32))
            ys.append(y)
            lens.append(len(X))
        got += len(fight)
        if got and got % 1000 < len(fight):
            print(f"{out_path}: {got}/{n_decisions}", flush=True)
    np.savez_compressed(out_path, X=np.concatenate(Xs), y=np.array(ys),
                        lens=np.array(lens))
    print(f"{out_path}: wrote {got} decisions", flush=True)


def _load_padded(paths, k_max=24):
    """(N, K, D) padded features, (N, K) mask, (N,) labels."""
    Xs, ys, lens = [], [], []
    for p in paths:
        blob = np.load(p)
        Xs.append(blob["X"]); ys.append(blob["y"]); lens.append(blob["lens"])
    X, y, lens = np.concatenate(Xs), np.concatenate(ys), np.concatenate(lens)
    D = X.shape[1]
    keep = []
    off = 0
    offs = []
    for i, L in enumerate(lens):
        if L <= k_max and y[i] < L:
            keep.append(i); offs.append(off)
        off += L
    N = len(keep)
    P = np.zeros((N, k_max, D), dtype=np.float32)
    M = np.zeros((N, k_max), dtype=bool)
    lab = np.zeros(N, dtype=np.int64)
    for j, (i, o) in enumerate(zip(keep, offs)):
        L = lens[i]
        P[j, :L] = X[o:o + L]
        M[j, :L] = True
        lab[j] = y[i]
    return P, M, lab


def train(out_json, paths, hidden=64, epochs=12, seed=0):
    import torch

    from deimos_bridge.neural_net import FEATURES, Net

    torch.manual_seed(seed)
    P, M, lab = _load_padded(paths)
    n_val = max(1, len(P) // 10)
    idx = np.random.RandomState(seed).permutation(len(P))
    val, trn = idx[:n_val], idx[n_val:]
    D = P.shape[2]
    net = torch.nn.Sequential(
        torch.nn.Linear(D, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Pt = torch.from_numpy(P)
    Mt = torch.from_numpy(M)
    yt = torch.from_numpy(lab)

    def batch_loss(ids):
        s = net(Pt[ids]).squeeze(-1)            # (B, K)
        s = s.masked_fill(~Mt[ids], -1e9)
        return torch.nn.functional.cross_entropy(s, yt[ids])

    def accuracy(ids):
        with torch.no_grad():
            s = net(Pt[ids]).squeeze(-1).masked_fill(~Mt[ids], -1e9)
            return float((s.argmax(1) == yt[ids]).float().mean())

    B = 512
    for ep in range(epochs):
        order = np.random.RandomState(seed + ep).permutation(trn)
        for i in range(0, len(order), B):
            opt.zero_grad()
            loss = batch_loss(torch.from_numpy(order[i:i + B]))
            loss.backward()
            opt.step()
        print(f"epoch {ep + 1}: teacher-match "
              f"train {accuracy(torch.from_numpy(trn[:4096])) * 100:.1f}% "
              f"val {accuracy(torch.from_numpy(val)) * 100:.1f}%",
              flush=True)

    layers = []
    for m in net:
        if isinstance(m, torch.nn.Linear):
            layers.append((m.weight.detach().numpy().T.tolist(),
                           m.bias.detach().numpy().tolist()))
    Net(layers).save(out_json)
    print(f"wrote {out_json} ({len(FEATURES)} features, "
          f"hidden {hidden})", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gen":
        generate(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
                 sys.argv[5] if len(sys.argv) > 5 else None)
    elif mode == "train":
        train(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(
            "usage: neural_bc.py gen N SEED OUT.npz [STUDENT.json] | "
            "train OUT.json IN.npz...")
