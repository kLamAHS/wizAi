"""Entity-encoded action scoring -- numpy end to end at inference.

The neural step the architecture notes gate on torch, taken the way
this repo takes dependencies: torch trains (`neural_bc.py`, repo
root), the weights ship as JSON, and this module runs the forward pass
in numpy -- the same split `leaf_value` proved out, so the live GUI
and the operator's machine never need torch installed.

The first neural seat is the rollout CONTINUATION, because that is
where the measurements point: the continuation is worth ~14 points of
kill rate, it is deck-specific (five hand-coded candidates are swept
per deck today), and it runs inside every rollout step -- so it must
be fast, which a ~10k-parameter MLP over entity features is, and it
needs no coverage of anything, which is what sank the tabular table.

Features are SCALE-FREE (fractions and ratios only), the lesson the
leaf value paid for: absolute health is what varies across worlds;
"this hit is 40% of the focus's health" means the same thing at level
5 and level 130. AUC 0.987 cross-school transfer came from exactly
this discipline.

Action model: every legal (card, target) pair -- plus passing -- is
scored by one MLP over [player context | board context | target |
card]; the policy plays the argmax. Deterministic on purpose: a
continuation that flips coins makes every rollout its own noise
source, and the sweeps that would pick it are paired precisely to
strip that noise out.

MEASURED. The shipped weights are two optimizers deep:

1. Behaviour cloning (81k teacher decisions) plus a 60k DAgger round
   (student drives, teacher labels), which fixed BC's collapse on hard
   boards (7.0 -> 17.5) and earned the first sweep seat: first on the
   storm deck's hard board on three seed streams of three
   (+5.0/+3.2/+2.6 over the best hand-coded, paired n=800).
2. Evolution strategies on top (neural_es.py): 25 generations of
   optimizing the SEAT's own objective -- paired win rate of
   greedy_ttk(6, continuation=net) -- from the BC seed, no torch, no
   teacher. Held out at n=800 paired against v1 and the hand-coded
   five: storm 700x2 29.9/31.5 (v1 18.1/18.4; best hand-coded 15.8)
   on two streams, low 700x2 22.4/22.1 (v1 7.5; best hand-coded 17.1)
   on two streams, low 600x2 65.5 (tied with the 65.8 leader, v1
   54.0), buffy 600x2 52.4 (second), buffy 700x2 9.1 -- LAST, against
   24.1. Deck-specific with a real weakness, which is exactly what
   the per-deck sweep routes around: where neural measures best it is
   installed, and on the buff-heavy deck it never will be.

TRIED AND REVERTED, recorded so it is not retried blind: a v2
featurizer added six hand-composition facts and two per-card
relatives (42 features; git f77d36a..f8a5739), on the theory that the
teacher's pip-banking decisions need to see the hand. It raised
teacher-match (68.1% -> 70.0% held out) and LOWERED seat strength --
the storm win fell from 18.1/18.4 (first on both streams) to
16.6/15.5 (first, then third), buffy 54.0 -> 49.9, same BC+DAgger
pipeline at the same scale. Mimicry accuracy and playing strength
are different objectives, and the extra capacity bought the former at
the latter's expense. A future variant should be judged in the seat
from the start; teacher-match is only a training diagnostic.
"""
import json
import os

import numpy as np

#: bump when the feature vector changes; a weights file trained on a
#: different layout must refuse to load rather than silently misread
FEATURE_VERSION = 1

#: feature names, in vector order -- the export guard checks these
FEATURES = (
    # player (8)
    "p_hp_frac", "p_norm_pips", "p_pow_pips", "p_eff_pips",
    "p_blade_mult", "p_hand_frac", "p_deck_frac", "p_incoming_frac",
    # board (4)
    "b_alive", "b_hp_vs_player", "b_focus_share", "b_dot_share",
    # target (8)
    "t_hp_frac", "t_hp_share", "t_school_mult", "t_ward_mult",
    "t_threat_frac", "t_dot_frac", "t_is_focus", "t_alive",
    # card (13)
    "c_damage", "c_drain", "c_blade", "c_trap", "c_shield", "c_heal",
    "c_debuff", "c_other", "c_pips", "c_dmg_vs_target", "c_dmg_vs_board",
    "c_accuracy", "c_aoe",
    # action (1)
    "is_pass",
)


def _charm_mult(actor, school):
    """Expected blade/weakness multiplier if a hit went out now --
    one charm per stacking identity, the sim's own rule."""
    mult, seen = 1.0, set()
    for h in actor.charms:
        if h.kind == "damage" and h.matches(school) \
                and h.stack_key not in seen:
            seen.add(h.stack_key)
            mult *= 1 + h.percent
    return mult


def _ward_mult(target, school):
    """Expected trap/shield multiplier for the next hit of `school`."""
    mult, seen = 1.0, set()
    for h in target.wards:
        if h.kind == "damage" and h.matches(school) \
                and h.stack_key not in seen:
            seen.add(h.stack_key)
            mult *= 1 + h.percent
    return mult


def _school_mult(target, school):
    res = target.resist.get(school, 0.0) + target.resist.get("*", 0.0)
    return (1 - res) * (1 + target.boost.get(school, 0.0))


def _dot_total(actor):
    return sum(o.per_tick * o.rounds_left
               for o in getattr(actor, "over_time", ())
               if o.kind == "dot")


def _clip(x, hi=4.0):
    return float(max(0.0, min(x, hi)))


def state_context(sim, s):
    """The [player | board] block, shared by every action this turn."""
    p = s.player
    school = sim.school
    alive = [e for e in s.enemies if e.alive]
    board_hp = sum(max(e.hp, 0.0) for e in alive) or 1.0
    incoming = sum(getattr(e, "flat_hit", 0.0) or 0.0 for e in alive)
    eff = p.norm_pips + 2 * p.pow_pips
    focus_hp = min((e.hp for e in alive), default=0.0)
    return np.array([
        _clip(p.hp / max(p.max_hp, 1.0), 1.0),
        _clip(p.norm_pips / 7.0, 1.0),
        _clip(p.pow_pips / 7.0, 1.0),
        _clip(eff / 14.0, 1.0),
        _clip(_charm_mult(p, school) - 1.0, 3.0) / 3.0,
        _clip(len(s.hand) / 8.0, 1.0),
        _clip(len(p.deck) / 24.0, 1.0),
        _clip(incoming / max(p.hp, 1.0), 2.0) / 2.0,
        _clip(len(alive) / 4.0, 1.0),
        _clip(board_hp / max(p.hp, 1.0)) / 4.0,
        _clip(focus_hp / board_hp, 1.0),
        _clip(sum(_dot_total(e) for e in alive) / board_hp, 1.0),
    ])


def target_block(sim, s, t):
    """The 8 target features for enemy index `t`."""
    school = sim.school
    alive = [e for e in s.enemies if e.alive]
    board_hp = sum(max(e.hp, 0.0) for e in alive) or 1.0
    e = s.enemies[t] if 0 <= t < len(s.enemies) else None
    if e is None or not e.alive:
        return np.zeros(8)
    focus_hp = min((x.hp for x in alive), default=0.0)
    return np.array([
        _clip(e.hp / max(e.max_hp, 1.0), 1.0),
        _clip(max(e.hp, 0.0) / board_hp, 1.0),
        _clip(_school_mult(e, school), 2.0) / 2.0,
        _clip(_ward_mult(e, school), 3.0) / 3.0,
        _clip((getattr(e, "flat_hit", 0.0) or 0.0)
              / max(s.player.hp, 1.0), 2.0) / 2.0,
        _clip(_dot_total(e) / max(e.hp, 1.0), 2.0) / 2.0,
        1.0 if e.hp == focus_hp else 0.0,
        1.0,
    ])


_KINDS = ("damage", "drain", "blade", "trap", "shield", "heal")


def card_block(sim, s, card, t):
    """The 13 card features, priced against target `t`."""
    school = sim.school
    p = s.player
    kind = np.zeros(8)
    if card.kind in _KINDS:
        kind[_KINDS.index(card.kind)] = 1.0
    elif card.kind in ("weakness", "mantle"):
        kind[6] = 1.0
    else:
        kind[7] = 1.0
    e = s.enemies[t] if 0 <= t < len(s.enemies) else None
    dmg = 0.0
    if card.kind in ("damage", "drain") and e is not None and e.alive:
        dmg = (card.damage * _school_mult(e, school)
               * _charm_mult(p, card.school) * _ward_mult(e, card.school))
    alive = [x for x in s.enemies if x.alive]
    board_hp = sum(max(x.hp, 0.0) for x in alive) or 1.0
    return np.concatenate([kind, [
        _clip(card.pips / 14.0, 1.0),
        _clip(dmg / max(e.hp, 1.0), 2.0) / 2.0 if e is not None else 0.0,
        _clip(dmg * (len(alive) if card.aoe else 1) / board_hp,
              2.0) / 2.0,
        _clip(card.accuracy, 1.0),
        1.0 if card.aoe else 0.0,
    ]])


def action_features(sim, s, card, t, ctx=None):
    ctx = state_context(sim, s) if ctx is None else ctx
    return np.concatenate([ctx, target_block(sim, s, t),
                           card_block(sim, s, card, t), [0.0]])


def pass_features(sim, s, ctx=None):
    ctx = state_context(sim, s) if ctx is None else ctx
    return np.concatenate([ctx, np.zeros(8), np.zeros(13), [1.0]])


def candidate_actions(sim, s):
    """[(card, target)] every castable pair, plus (None, None) = pass.

    The same enumeration greedy_ttk searches over, so the net and the
    search are choosing from the same menu.
    """
    from .policies import aimed_at_one_enemy

    foes = [i for i, e in enumerate(s.enemies) if e.alive] or [0]
    out, seen = [], set()
    for card in s.hand:
        if card.name in seen:
            continue
        aims = foes if aimed_at_one_enemy(card) else foes[:1]
        playable = [t for t in aims if sim.can_cast(s, card, t)]
        if playable:
            seen.add(card.name)
            out.extend((card, t) for t in playable)
    out.append((None, None))
    return out


def decision_matrix(sim, s):
    """(candidates, feature matrix) for one decision -- the shared
    path between data generation, training and inference."""
    cands = candidate_actions(sim, s)
    ctx = state_context(sim, s)
    rows = [pass_features(sim, s, ctx) if card is None
            else action_features(sim, s, card, t, ctx)
            for card, t in cands]
    return cands, np.stack(rows)


# ---------------------------------------------------------------- the net

class Net:
    """A plain relu MLP over action features; numpy forward only."""

    def __init__(self, layers):
        #: [(W, b)] with W shaped (in, out)
        self.layers = [(np.asarray(W, dtype=float),
                        np.asarray(b, dtype=float)) for W, b in layers]

    def scores(self, X):
        h = np.asarray(X, dtype=float)
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h[:, 0]

    def save(self, path):
        json.dump({"feature_version": FEATURE_VERSION,
                   "features": list(FEATURES),
                   "layers": [{"W": W.tolist(), "b": b.tolist()}
                              for W, b in self.layers]},
                  open(path, "w"))

    @classmethod
    def load(cls, path):
        blob = json.load(open(path))
        if blob.get("feature_version") != FEATURE_VERSION or \
                list(blob.get("features") or []) != list(FEATURES):
            raise ValueError(
                "weights were trained on a different feature layout -- "
                "retrain (neural_bc.py) rather than misread them")
        return cls([(L["W"], L["b"]) for L in blob["layers"]])


DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "neural_continuation.json")
#: the second seat: the campaign-4 charm specialist ("neural-b" in
#: CONTINUATIONS). Separate weights, same architecture, priced by the
#: same per-deck sweep -- more coverage means more seats, not a wider
#: objective on one net.
DEFAULT_WEIGHTS_B = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "neural_continuation_b.json")
#: the third seat: the campaign-5 sustain specialist ("neural-c"),
#: shipped on the replicated L30 death win (34.9/38.1 held out vs
#: 28.5/31.8 for school-aware(3), the previous leader there)
DEFAULT_WEIGHTS_C = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "neural_continuation_c.json")


def net_policy(path=None):
    """A policy/continuation closure over the shipped weights.

    Raises if the weights are missing or stale -- callers that want a
    quiet fallback (`build_continuation`) catch and keep the default.
    """
    net = Net.load(path or DEFAULT_WEIGHTS)

    def strat(sim, s):
        cands, X = decision_matrix(sim, s)
        card, t = cands[int(np.argmax(net.scores(X)))]
        return None if card is None else (card, t)

    return strat
