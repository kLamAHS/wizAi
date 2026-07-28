"""
Recurrent (sequence) policy: hand-written BPTT correctness, linear
equivalence at init, per-fight state reset, and episode time order.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import load_spells_full, LIVE_RULES
from generalist import FEATS, GeneralistQ
from seq_policy import (RecurrentQ, episodes_from, train_seq_bc,
                        _softmax_masked)
from w101_sim import Sim, Boss, evaluate_paired, make_blade_stack

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))
F = len(FEATS)


def _episode(T=5, A=4, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.5, (T, A, F))
    M = np.ones((T, A), bool)
    M[:, -1] = False                     # one illegal slot everywhere
    C = rng.integers(0, A - 1, T)
    return P, M, C


def test_bptt_matches_numerical_gradient():
    """The reason every number from this module is trustworthy."""
    model = RecurrentQ(H=4, seed=1)
    rng = np.random.default_rng(2)
    for k in model.p:                    # non-degenerate parameters
        model.p[k] = rng.normal(0, 0.3, model.p[k].shape)
    P, M, C = _episode(T=6, A=5, seed=3)
    loss0, grads = model.episode_grads(P, M, C)

    def loss_of(params):
        keep = model.p
        model.p = params
        val = model.episode_grads(P, M, C)[0]
        model.p = keep
        return val

    eps = 1e-6
    for k, g in grads.items():
        flat = model.p[k].reshape(-1)
        picks = rng.choice(flat.size, size=min(6, flat.size),
                           replace=False)
        for i in picks:
            up = {kk: v.copy() for kk, v in model.p.items()}
            dn = {kk: v.copy() for kk, v in model.p.items()}
            up[k].reshape(-1)[i] += eps
            dn[k].reshape(-1)[i] -= eps
            num = (loss_of(up) - loss_of(dn)) / (2 * eps)
            ana = g.reshape(-1)[i]
            assert abs(num - ana) < 1e-4 * max(1.0, abs(num)), \
                (k, i, num, ana)


def test_zero_U_is_exactly_the_linear_policy():
    w = np.random.default_rng(4).normal(0, 0.5, F)
    lin = GeneralistQ(w=w)
    seq = RecurrentQ.from_linear(w, H=6)
    boss = Boss("dummy", 900, "death", 0)
    boss.resist_map = {"death": 0.5}
    sim1 = Sim(dict(CARDS), ["Wraith", "Wraith", "Deathblade", "Feint"],
               "death", boss, player_hp=10**9, rules=LIVE_RULES)
    sim2 = Sim(dict(CARDS), ["Wraith", "Wraith", "Deathblade", "Feint"],
               "death", boss, player_hp=10**9, rules=LIVE_RULES)
    a = evaluate_paired(sim1, {"p": lin.policy()}, n=120)["p"]
    b = evaluate_paired(sim2, {"p": seq.policy()}, n=120)["p"]
    assert a["win_rate"] == b["win_rate"]
    assert (a["mean_ttk"] == b["mean_ttk"] or
            (a["mean_ttk"] != a["mean_ttk"] and
             b["mean_ttk"] != b["mean_ttk"]))


def test_hidden_state_does_not_leak_between_fights():
    """One policy object replaying the SAME seeded fight twice must
    produce the same line — otherwise fight 1's plan contaminates
    fight 2 and evaluation order would change the results."""
    seq = RecurrentQ(H=4, seed=5)
    rng = np.random.default_rng(6)
    seq.p["U"] = rng.normal(0, 0.6, (F, 4))
    seq.p["w0"] = rng.normal(0, 0.4, F)
    boss = Boss("dummy", 900, "death", 0)
    boss.resist_map = {"death": 0.5}
    deck = ["Wraith", "Wraith", "Deathblade", "Feint", "Banshee"]
    pol = seq.policy()

    def play(seed):
        sim = Sim(dict(CARDS), deck, "death", boss, player_hp=10**9,
                  rules=LIVE_RULES, rng=random.Random(seed))
        s = sim.new_state()
        line = []
        for _ in range(6):
            a = pol(sim, s)
            line.append(None if a is None else
                        (a[0].name if isinstance(a, tuple) else a.name))
            if a is not None:
                if isinstance(a, tuple):
                    sim.cast(s, a[0], target=a[1])
                else:
                    sim.cast(s, a)
            if not any(e.alive for e in s.enemies):
                break
            sim.end_round(s)
        return line

    first = play(11)
    play(99)                              # a different fight in between
    assert play(11) == first

    # and within a fight the hidden state genuinely moves the ranking
    h1 = seq.step_h(np.zeros(4), np.ones(F) * 0.3)
    assert np.any(np.abs(seq.w_eff(h1) - seq.p["w0"]) > 1e-6)


def test_episodes_flip_to_forward_time():
    """Generators log steps in reverse (return-to-go accumulates
    backwards); the recurrence would train backwards without the
    flip."""
    T = 4
    P = np.arange(T * 2 * F, dtype=float).reshape(T, 2, F)
    C = np.array([0, 1, 1, 0], np.int16)          # per-step labels
    data = dict(phis=P[::-1].copy(), mask=np.ones((T, 2), bool),
                chosen=C[::-1].copy(),            # logged in reverse,
                G=np.zeros(T, np.float32),        # like the generators
                won=np.ones(T, bool), ep=np.zeros(T, np.int32))
    (Pe, Me, Ce), = episodes_from(data)
    assert np.allclose(Pe, P)             # forward order restored
    assert np.array_equal(Ce, C)          # and labels move WITH it
    # features and labels must stay aligned step by step
    for t in range(T):
        assert np.allclose(Pe[t], P[t]) and Ce[t] == C[t]


def test_train_seq_bc_runs_and_improves_fit():
    """A tiny synthetic set the linear class cannot fit but the
    recurrent one can: the right action alternates by step parity."""
    rng = np.random.default_rng(7)
    eps_data = {"phis": [], "mask": [], "chosen": [], "G": [],
                "won": [], "ep": []}
    base = rng.normal(0, 1.0, (2, F))
    for e in range(40):
        T = 6
        P = np.repeat(base[None, :, :], T, axis=0)
        C = np.array([t % 2 for t in range(T)], np.int16)
        # log in reverse, like the generators do
        eps_data["phis"].append(P[::-1])
        eps_data["mask"].append(np.ones((T, 2), bool))
        eps_data["chosen"].append(C[::-1])
        eps_data["G"].append(np.zeros(T, np.float32))
        eps_data["won"].append(np.ones(T, bool))
        eps_data["ep"].append(np.full(T, e, np.int32))
    data = {k: np.concatenate(v) for k, v in eps_data.items()}
    model = train_seq_bc(data, H=6, epochs=25, lr=0.05, batch=8, seed=0)
    P, M, C = episodes_from(data)[0]
    h = np.zeros(model.H)
    right = 0
    for t in range(len(C)):
        q = P[t] @ model.w_eff(h)
        right += int(np.argmax(q) == C[t])
        h = model.step_h(h, P[t][C[t]])
    assert right >= len(C) - 1            # parity learned from memory


def test_training_never_mutates_the_caller_weights():
    """Regression: np.asarray ALIASES, so from_linear(w) shared the
    array with the linear baseline and Adam's in-place update silently
    retrained it — the reported 'linear' column was the recurrent
    model's w0. Caught by adversarial review, not by any assertion."""
    rng = np.random.default_rng(9)
    w_lin = rng.normal(0, 0.3, F)
    w_before = w_lin.copy()
    lin = GeneralistQ(w=w_lin)
    lin_before = lin.w.copy()
    data = {"phis": [], "mask": [], "chosen": [], "G": [], "won": [],
            "ep": []}
    for e in range(6):
        T, A = 4, 3
        data["phis"].append(rng.normal(0, 0.5, (T, A, F)))
        data["mask"].append(np.ones((T, A), bool))
        data["chosen"].append(rng.integers(0, A, T).astype(np.int16))
        data["G"].append(np.zeros(T, np.float32))
        data["won"].append(np.ones(T, bool))
        data["ep"].append(np.full(T, e, np.int32))
    data = {k: np.concatenate(v) for k, v in data.items()}
    model = train_seq_bc(data, w_linear=w_lin, H=4, epochs=3, lr=0.05,
                         batch=3, seed=0)
    assert np.array_equal(w_lin, w_before), "caller's array mutated"
    assert np.array_equal(lin.w, lin_before), "baseline model mutated"
    assert not np.array_equal(model.p["w0"], w_before)   # it did train


def test_softmax_masked_survives_no_legal_actions():
    from seq_policy import _softmax_masked
    p = _softmax_masked(np.array([1.0, 2.0]), np.array([False, False]))
    assert np.all(np.isfinite(p)) and abs(p.sum() - 1.0) < 1e-12


def test_softmax_masked_ignores_illegal():
    q = np.array([1.0, 5.0, 2.0])
    m = np.array([True, False, True])
    p = _softmax_masked(q, m)
    assert p[1] == 0.0
    assert abs(p.sum() - 1.0) < 1e-12
