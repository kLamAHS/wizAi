"""The neural seat's torch-free half: features, forward pass, weights
guard, and the policy contract. Training lives in neural_bc.py and
needs torch; nothing here does -- which is the point of the split.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import load_spells_full
from deimos_bridge.neural_net import (FEATURES, Net, decision_matrix,
                                      net_policy)
from w101_sim import Boss, Sim

CARDS = load_spells_full()


def _sim():
    return Sim(CARDS, ["Frost Beetle"] * 3 + ["Ice Trap"] * 2
               + ["Snow Serpent"] * 3, "ice",
               Boss(name="b", hp=690, school="death", dmg=90),
               enemies=[Boss(name="m", hp=300, school="fire", dmg=60)],
               player_hp=1022)


def test_features_are_bounded_and_the_menu_matches_the_search():
    sim = _sim()
    s = sim.new_state()
    cands, X = decision_matrix(sim, s)
    assert X.shape == (len(cands), len(FEATURES))
    assert np.isfinite(X).all()
    assert X.min() >= 0.0 and X.max() <= 1.0
    # passing is always on the menu, exactly once, at the end
    assert cands[-1] == (None, None)
    assert sum(1 for c, _ in cands if c is None) == 1
    # single-enemy cards get a row per living enemy
    beetles = [t for c, t in cands if c is not None
               and c.name == "Frost Beetle"]
    assert sorted(beetles) == [0, 1]


def test_the_forward_pass_is_the_arithmetic_it_claims():
    net = Net([([[1.0], [2.0]], [0.5]),      # 2 -> 1, relu skipped (last)
               ])
    got = net.scores(np.array([[1.0, 1.0], [0.0, 2.0]]))
    assert np.allclose(got, [3.5, 4.5])
    # two layers: relu between
    net2 = Net([([[1.0], [-1.0]], [0.0]), ([[2.0]], [1.0])])
    got2 = net2.scores(np.array([[1.0, 2.0]]))   # relu(-1) = 0 -> 1.0
    assert np.allclose(got2, [1.0])


def test_stale_weights_refuse_to_load(tmp_path):
    import json

    p = tmp_path / "w.json"
    rng = np.random.RandomState(0)
    net = Net([(rng.randn(len(FEATURES), 8), rng.randn(8)),
               (rng.randn(8, 1), rng.randn(1))])
    net.save(str(p))
    again = Net.load(str(p))
    assert all(np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
               for a, b in zip(net.layers, again.layers))

    blob = json.load(open(p))
    blob["features"] = ["something", "else"]
    json.dump(blob, open(p, "w"))
    try:
        Net.load(str(p))
        raise AssertionError("stale feature layout loaded silently")
    except ValueError:
        pass


def test_the_net_policy_plays_a_whole_fight_legally(tmp_path):
    """Random weights, real fight: every move the net returns must be
    castable, and the fight must complete without the policy breaking
    the sim -- the same contract every continuation honours."""
    rng = np.random.RandomState(7)
    p = tmp_path / "w.json"
    Net([(rng.randn(len(FEATURES), 16) * 0.3, rng.randn(16) * 0.1),
         (rng.randn(16, 1) * 0.3, rng.randn(1) * 0.1)]).save(str(p))
    pol = net_policy(str(p))

    sim = _sim()
    s = sim.new_state()
    move = pol(sim, s)
    assert move is None or sim.can_cast(s, move[0], move[1])
    t, won, _ = sim.run(pol, max_turns=20)
    assert t >= 1


def test_missing_weights_raise_rather_than_guess(tmp_path):
    try:
        net_policy(str(tmp_path / "nope.json"))
        raise AssertionError("loaded weights that do not exist")
    except (OSError, ValueError):
        pass


def test_neural_holds_a_sweep_seat():
    """Both net seats: in the sweep, buildable, installable -- so
    choose_search prices them per deck like the hand-coded five. And
    genuinely DIFFERENT nets: two seats with one behaviour would be
    the redundancy campaign 2 rejected."""
    import numpy as np

    from deimos_bridge import neural_net, policies as P

    for name in ("neural", "neural-b", "neural-c"):
        assert name in P.CONTINUATIONS
        pol = P.build_continuation(name)
        sim = _sim()
        s = sim.new_state()
        move = pol(sim, s)
        assert move is None or sim.can_cast(s, move[0], move[1])
        assert P.set_continuation(name) == name
    P.set_continuation(P.DEFAULT_CONTINUATION)

    nets = [neural_net.Net.load(p) for p in
            (neural_net.DEFAULT_WEIGHTS, neural_net.DEFAULT_WEIGHTS_B,
             neural_net.DEFAULT_WEIGHTS_C)]
    for i in range(len(nets)):
        for j in range(i + 1, len(nets)):
            assert not all(np.allclose(x[0], y[0])
                           for x, y in zip(nets[i].layers,
                                           nets[j].layers))


def test_unloadable_weights_cost_the_candidate_not_the_sweep(monkeypatch):
    """A missing or stale weights file must never break a sweep or a
    live fight -- build_continuation("neural") quietly hands back the
    default heuristic instead."""
    from deimos_bridge import neural_net, policies as P

    monkeypatch.setattr(neural_net, "DEFAULT_WEIGHTS",
                        "/nonexistent/weights.json")
    pol = P.build_continuation("neural")
    sim = _sim()
    s = sim.new_state()
    move = pol(sim, s)                     # the fallback still plays
    assert move is None or sim.can_cast(s, move[0], move[1])
