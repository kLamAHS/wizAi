"""
Treasure cards in the action space: the draw reflex, same-round cast,
the no-discard rule, and both agents playing from the sideboard.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w101_sim import (Sim, Boss, load_cards, evaluate, make_blade_stack,
                      with_tc_draw)
from rl_agent import QAgent
from generalist import GeneralistQ, FEATS

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_cards(str(ROOT / "cards_clean.json"))


def _sim(sideboard, hp=10**9, boss_hp=1000, dmg=0):
    b = Boss("dummy", boss_hp, "death", dmg)
    return Sim(dict(CARDS), ["Fire Cat"] * 4, "fire", b, player_hp=hp,
               sideboard=sideboard, rng=random.Random(0))


def test_tc_reflex_draws_and_is_castable_same_round():
    sim = _sim(["Helephant@tc", "Helephant@tc"])
    s = sim.new_state()
    n0 = len(s.player.sideboard)
    pol = with_tc_draw(lambda sim, s: None)
    pol(sim, s)
    assert len(s.player.sideboard) == n0 - 1
    tc = next(c for c in s.hand if c.source == "tc")
    assert tc.name == "Helephant@tc"
    assert tc in s.player.fresh_tc
    assert sim.discard(s, tc) is False      # game rule: fresh TC stays
    s.norm_pips, s.pow_pips = 6, 0
    assert sim.can_cast(s, tc)              # but it can be cast now


def test_qagent_learns_to_cast_the_tc_nuke():
    """4x Fire Cat cannot kill 1000 HP; the sideboard Helephants can —
    a trained agent must find them."""
    sim = _sim(["Helephant@tc", "Helephant@tc", "Helephant@tc"])
    agent = QAgent(dict(CARDS), ["Fire Cat"] * 4, "fire",
                   rng=random.Random(1))
    for ep in range(2500):
        frac = ep / 2500
        agent.alpha = 0.30 * (1 - 0.9 * frac)
        agent.train_episode(sim, eps=max(0.05, 0.3 * (1 - frac)), dp_w=0.0)
    w, m = evaluate(sim, agent.policy(), n=400)
    assert w > 0.6, (w, m)


def test_generalist_casts_tcs_zero_shot():
    """The linear policy values cards by properties, so a TC nuke in
    hand needs no retraining — overkill/kill_now light up for it."""
    gen = GeneralistQ()
    gen.w[FEATS.index("overkill")] = 1.0
    gen.w[FEATS.index("kill_now")] = 1.0
    sim = _sim(["Helephant@tc", "Helephant@tc", "Helephant@tc"])
    w, m = evaluate(sim, gen.policy(), n=300)
    assert w > 0.5, (w, m)


def test_tc_casts_are_audited():
    """TCs cost real gold: every fight counts them, every paired eval
    reports them — a policy leaning on TCs can't hide it."""
    from w101_sim import evaluate_paired
    gen = GeneralistQ()
    gen.w[FEATS.index("overkill")] = 1.0
    sim = _sim(["Helephant@tc"] * 3)
    st = evaluate_paired(sim, {"gen": gen.policy()}, n=50)["gen"]
    assert "mean_tc_casts" in st and st["mean_tc_casts"] > 0
    # and a TC-free fight audits to exactly zero
    sim0 = _sim(None, boss_hp=300)
    st0 = evaluate_paired(sim0, {"h": make_blade_stack(0)}, n=50)["h"]
    assert st0["mean_tc_casts"] == 0.0


def test_no_sideboard_behavior_unchanged():
    """Without a sideboard the reflex is inert: blade-stack play on a
    plain deck is byte-identical to the pre-TC path."""
    sim = _sim(None, boss_hp=300)
    w, m = evaluate(sim, make_blade_stack(0), n=300)
    sim2 = _sim(None, boss_hp=300)
    w2, m2 = evaluate(sim2, with_tc_draw(make_blade_stack(0)), n=300)
    assert (w, m) == (w2, m2)
