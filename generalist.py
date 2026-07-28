"""
Deck-conditioned combat policy: bilevel rung 4.

The per-deck QAgent keys its table on card IDENTITIES, so every
candidate deck needs its own 8k-episode fine-tune — the cost center of
build_deck's stage 2. This generalist replaces the table with a linear
Q over card-vs-state FEATURES (what kind of card, how hard it hits
this boss through the buffs that are up, whether it kills now, whether
its blade is a duplicate...), so one policy plays ANY legal deck
zero-shot: deck-conditioning enters through the features of the cards
actually in hand.

Trained with the same backward Monte-Carlo returns as the QAgent, but
each episode samples a fresh random (school, deck, boss) — the policy
that emerges is the RULE (blade into cash-in, wait on X-pip, don't
re-blade), not a memorized line. Honesty notes: est_dmg is a crude
model (flat +30%/blade, +25%/trap, no prism conversion) — it is a
FEATURE the regression can reweight, not the simulator; and the
held-out comparison in __main__ reports the generalist against both
the scripted heuristic floor and the per-deck fine-tune ceiling.
"""
import json
import random

import numpy as np

from w101_sim import Sim, OPPOSING, evaluate, tc_reflex
from rl_agent import PASS, MAX_TURNS, FAIL_PENALTY, _dig, DMG_KINDS

FEATS = [
    "bias", "hp_frac", "pip_frac", "hits_left",
    "k_hit", "k_blade", "k_trap", "k_prism", "k_heal", "k_shield",
    "k_weakness", "k_pass",
    "acc", "dmg_est", "overkill", "kill_now",
    "hit_x_blades", "hit_x_traps", "blade_x_hitsleft", "dup",
    "prism_gain", "xpip_x_pips", "pass_x_lowpips", "heal_x_deficit",
    "buff_pct", "pass_x_xpip_wait",
    # target dimension (multi-enemy actions only; zero in 1v1 so old
    # weight vectors stay valid — load() zero-pads them)
    "tgt_hp_frac", "tgt_is_support",
]
G_SCALE = 10.0        # returns live in [-65, -1]; scale for SGD stability


def _buff_factors(s, school, tgt=None):
    """Exact hanging-effect multipliers for a hit of `school` (the sim's
    own math is richer — prisms, orders — but percents beat counts).
    `tgt` selects whose wards count; the primary enemy keeps the
    original s.traps view so 1v1 vectors are unchanged."""
    blade = 1.0
    for h in s.blades.values():
        if h.kind == "damage" and (h.schools is None or school in h.schools):
            blade *= 1 + h.percent
    if tgt is None or tgt is s.enemies[0]:
        traps = s.traps.values()
    else:
        traps = [(h, h.charges) for h in tgt.wards
                 if h.kind == "prism" or
                 (h.kind == "damage" and h.percent > 0)]
    trap = 1.0
    for h, n in traps:
        if h.kind == "damage" and (h.schools is None or school in h.schools):
            trap *= (1 + h.percent) ** n
    return blade, trap


def _tgt_mult(sim, s, school, tgt):
    """Incoming multiplier for `school` damage on `tgt`. The primary
    enemy keeps the boss-template path (bit-exact with pre-target
    vectors); other actors use the sim's real resist math."""
    if tgt is None or tgt is s.enemies[0]:
        return sim.boss.incoming_mult(school)
    return sim._resist_mult(tgt, school, 0.0)


def _est_damage(sim, s, c, blade_f, trap_f, mult):
    """Expected-damage model for the feature vector only."""
    if c.x_pips:
        pow_v = 2 if c.school == sim.school else 1
        base = c.damage * (s.norm_pips + pow_v * s.pow_pips)
    else:
        base = c.damage
    return base * mult * blade_f * trap_f


SUPPORT = ("healer", "buffer", "debuffer")
FOE_KINDS = ("damage", "drain", "trap", "weakness")


def phi(sim, s, a):
    """Feature vector for action a: PASS, a Card (aimed at the primary
    enemy), or (Card, target_idx) in multi-enemy fights. Card-only
    vectors are bit-identical to the pre-target implementation."""
    tgt = None
    if isinstance(a, tuple):
        a, ti = a
        tgt = s.enemies[ti]
    x = np.zeros(len(FEATS))
    pipf = min((s.norm_pips + 2 * s.pow_pips +
                2 * getattr(s, 'school_pips', 0)) / 14.0, 1.0)
    hits_left = min(sum(1 for c in s.hand if c.kind in DMG_KINDS) +
                    sum(1 for c in s.deck if c.kind in DMG_KINDS), 8) / 8.0
    x[0] = 1.0
    x[1] = min(s.boss_hp / 3000.0, 3.0)
    x[2] = pipf
    x[3] = hits_left
    if a == PASS:
        x[11] = 1.0
        x[22] = 1.0 - pipf
        if any(c.x_pips and c.kind in DMG_KINDS for c in s.hand):
            x[25] = 1.0 - pipf      # building pips for a waiting X-pip
        return x
    c = a
    tgt_hp = tgt.hp if tgt is not None else s.boss_hp
    x[12] = c.accuracy
    if tgt is not None and tgt is not s.enemies[0]:
        x[26] = min(tgt_hp / 3000.0, 3.0)
        x[27] = 1.0 if (getattr(tgt, "archetype", "") in SUPPORT and
                        getattr(tgt, "spell_pool", None) is not None) \
            else 0.0
    if c.kind in DMG_KINDS:
        x[4] = 1.0
        blade_f, trap_f = _buff_factors(s, c.school, tgt)
        est = _est_damage(sim, s, c, blade_f, trap_f,
                          _tgt_mult(sim, s, c.school, tgt))
        x[13] = min(est / 1000.0, 3.0)
        x[14] = min(est / max(tgt_hp, 1), 3.0)
        x[15] = 1.0 if est >= tgt_hp else 0.0
        x[16] = min(blade_f - 1.0, 2.0) / 2.0
        x[17] = min(trap_f - 1.0, 2.0) / 2.0
        if c.x_pips:
            x[21] = pipf
    elif c.kind == "blade":
        x[5] = 1.0
        x[18] = 1.0 if hits_left > 0 else 0.0
        x[19] = 1.0 if c.name in s.blades else 0.0
        x[24] = c.percent
    elif c.kind in ("trap", "weakness"):
        x[6 if c.kind == "trap" else 10] = 1.0
        if tgt is not None and tgt is not s.enemies[0]:
            lst = tgt.wards if c.kind == "trap" else tgt.charms
            x[19] = 1.0 if any(h.name == c.name for h in lst) else 0.0
        else:
            x[19] = 1.0 if c.name in s.traps else 0.0
        x[24] = abs(c.percent)
    elif c.kind == "prism":
        x[7] = 1.0
        opp = OPPOSING.get(sim.school)
        if opp:
            x[20] = sim.boss.incoming_mult(opp) - \
                sim.boss.incoming_mult(sim.school)
        x[19] = 1.0 if c.name in s.traps else 0.0
    elif c.kind == "heal":
        x[8] = 1.0
        x[23] = 1.0 - min(s.player_hp / max(sim.player_hp0, 1), 1.0)
    elif c.kind == "shield":
        x[9] = 1.0
    return x


def act_key(x):
    """Normalize an action for identity matching: focus-style wrappers
    tuple EVERY card, but legal_actions only targets foe-directed
    kinds, so a teacher's bladed tuple must match the bare blade."""
    if x is None or x == PASS:
        return ("__pass__", None)
    if isinstance(x, tuple):
        card, tgt = x
        return (card.name, tgt if card.kind in FOE_KINDS else None)
    return (x.name, None)


def match_action(acts, a):
    """Index of `a` in the action list, PASS index 0 as fallback."""
    want = act_key(a)
    for i, act in enumerate(acts):
        if act_key(act) == want:
            return i
    return 0


def legal_actions(sim, s):
    """PASS plus one instance of each distinct castable card. In
    multi-enemy fights, foe-directed kinds expand to one action per
    living target; 1v1 keeps the bare-card actions (and therefore the
    exact behavior) of the pre-target implementation."""
    acts, seen = [PASS], set()
    multi = len(s.living_enemies()) > 1
    for c in s.hand:
        if c.name in seen:
            continue
        if multi and c.kind in FOE_KINDS:
            added = False
            for i, e in enumerate(s.enemies):
                if e.alive and sim.can_cast(s, c, i):
                    acts.append((c, i))
                    added = True
            if added:
                seen.add(c.name)
        elif sim.can_cast(s, c):
            seen.add(c.name)
            acts.append(c)
    return acts


class GeneralistQ:
    """Linear Q over card-vs-state features; plays any deck zero-shot."""

    def __init__(self, alpha=0.03, rng=None, w=None):
        # np.array, not asarray: never alias a caller's weights
        self.w = np.zeros(len(FEATS)) if w is None else np.array(w, float)
        # state-value baseline head for training: [bias, boss-hp frac,
        # pip frac, in-a-mob-fight]. Features that fire only in HARD
        # episodes (tgt_is_support exists only in mob fights, and mob
        # fights mostly lose) otherwise absorb the difficulty as
        # negative weight — the credit-confound cousin of "losers
        # teach losing habits" from the offline-RL rung.
        self.v = np.zeros(4)
        self.alpha = alpha
        self.rng = rng or random.Random()

    def q(self, sim, s, a):
        return float(self.w @ phi(sim, s, a))

    def act(self, sim, s, eps):
        legal = legal_actions(sim, s)
        if self.rng.random() < eps:
            return self.rng.choice(legal)
        return max(legal, key=lambda a: self.q(sim, s, a))

    def train_episode(self, sim, eps, advisor=None, adv_w=0.0):
        """Backward MC on shared weights (same scheme as the QAgent).
        Win = ALL enemies dead (matters in mob fights). `advisor` is a
        scripted policy followed with probability adv_w — guided
        exploration that puts completed multi-step sequences (e.g. a
        finished healer kill) into the data where eps-greedy alone
        almost never lands them."""
        s = sim.new_state()
        traj, won = [], False
        while True:
            tc_reflex(sim, s)       # TC reflex: draw free, cast learned
            a = None
            if advisor is not None and self.rng.random() < adv_w:
                a = advisor(sim, s)
                if a is None:
                    a = PASS
                elif isinstance(a, tuple) and \
                        a[0].kind not in FOE_KINDS:
                    a = a[0]        # focus wraps everything; only
                                    # foe-directed kinds keep targets
            if a is None:
                a = self.act(sim, s, eps)
            sv = np.array([1.0, min(s.boss_hp / 3000.0, 3.0),
                           min((s.norm_pips + 2 * s.pow_pips) / 14.0, 1.0),
                           1.0 if len(s.living_enemies()) > 1 else 0.0])
            traj.append((phi(sim, s, a), sv))
            if a == PASS:
                _dig(s)
            elif isinstance(a, tuple):
                sim.cast(s, a[0], target=a[1])
            else:
                sim.cast(s, a)
            if not any(e.alive for e in s.enemies):
                won = True
                turns = s.turn + 1
                break
            sim.end_round(s)
            if s.player_hp <= 0 or s.turn >= MAX_TURNS:
                turns = s.turn
                break
        G = 0.0 if won else -FAIL_PENALTY
        for x, sv in reversed(traj):
            G = -1.0 + G
            g = G / G_SCALE
            base = float(self.v @ sv)
            self.v += self.alpha * (g - base) * sv
            self.w += self.alpha * ((g - base) - float(self.w @ x)) * x
        return turns, won

    def policy(self):
        def pol(sim, s):
            tc_reflex(sim, s)
            a = max(legal_actions(sim, s),
                    key=lambda a: self.q(sim, s, a))
            if a == PASS:
                _dig(s)
                return None
            return a               # Card, or (Card, target) tuple —
        return pol                 # Sim.run normalizes both

    def save(self, path):
        json.dump({"features": FEATS, "w": self.w.tolist()},
                  open(path, "w", encoding="utf-8"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        w = list(d["w"])
        if len(w) < len(FEATS):    # pre-target policy: new features
            w += [0.0] * (len(FEATS) - len(w))   # are simply ignored
        return cls(w=w)


def _rand_support(rng):
    """Random living support minion for mob training episodes."""
    from w101_sim import Boss
    m = Boss(f"minion-{rng.randrange(9999)}", rng.randrange(200, 500),
             "life", 0, pool=["Sprite", "Pixie"], archetype="healer")
    m.resist_map = {}
    m.boost_map = {}
    return m


def train_generalist(cards, episodes=40000, seed=0, rules=None,
                     log=None, snap_every=5000, val_pairs=None,
                     mob_frac=0.25, advisor=None):
    """Each episode draws a fresh random (school, deck, boss); a
    `mob_frac` slice adds a living healer minion so the target
    dimension gets trained. Snapshots keep the weights that score
    best on a FIXED validation set (which includes one mob pair) so
    late exploration can't corrupt the returned policy."""
    from deck_builder import legal_pool, sample_deck, random_boss

    rng = random.Random(seed)
    agent = GeneralistQ(rng=random.Random(seed + 1))
    schools = ("fire", "ice", "storm", "myth", "life", "death", "balance")
    pools = {sc: legal_pool(cards, sc) for sc in schools}
    if val_pairs is None:
        vrng = random.Random(seed + 2)
        val_pairs = []
        for i in range(5):
            sc = schools[i % len(schools)]
            b = random_boss(vrng, f"val{i}")
            dl = sample_deck(pools[sc], sc, b, vrng)
            val_pairs.append((sc, dl, b, []))
        if mob_frac > 0:
            b = random_boss(vrng, "valmob")
            dl = sample_deck(pools["death"], "death", b, vrng)
            val_pairs.append(("death", dl, b, [_rand_support(vrng)]))
    best = (-1.0, None)
    for ep in range(episodes):
        frac = ep / episodes
        agent.alpha = 0.03 * (1 - 0.9 * frac)
        sc = schools[rng.randrange(len(schools))]
        boss = random_boss(rng, f"train{ep}")
        dl = sample_deck(pools[sc], sc, boss, rng)
        if dl is None:
            continue
        extra = [_rand_support(rng)] if rng.random() < mob_frac else []
        sim = Sim(dict(cards), dl, sc, boss, player_hp=10**9,
                  rules=rules, rng=rng, enemies=extra)
        agent.train_episode(sim, eps=max(0.05, 0.3 * (1 - frac)),
                            advisor=advisor,
                            adv_w=max(0.0, 0.5 * (1 - 2 * frac)))
        if (ep + 1) % snap_every == 0:
            wins = []
            for sc_v, dl_v, b_v, ex_v in val_pairs:
                simv = Sim(dict(cards), dl_v, sc_v, b_v, player_hp=10**9,
                           rules=rules, enemies=ex_v)
                w, m = evaluate(simv, agent.policy(), n=300)
                wins.append(w)
            score = sum(wins) / len(wins)
            if score >= best[0]:
                best = (score, agent.w.copy())
            if log:
                print(f"    ep {ep+1:>6}: val kill "
                      f"{score*100:5.1f}%  (best {best[0]*100:5.1f}%)",
                      flush=True)
    if best[1] is not None:
        agent.w = best[1]
    return agent


if __name__ == "__main__":
    import time

    from data_full import load_spells_full, LIVE_RULES
    from deck_builder import (legal_pool, sample_deck, random_boss,
                              build_deck, fine_tune)
    from w101_sim import make_blade_stack

    cards = load_spells_full()
    t0 = time.time()
    print("[1/3] training the generalist: 40k episodes, fresh random "
          "(school, deck, boss) each episode", flush=True)
    gen = train_generalist(cards, episodes=40000, seed=0,
                           rules=LIVE_RULES, log=True)
    gen.save("generalist.json")
    print(f"    trained + saved generalist.json ({time.time()-t0:.0f}s)",
          flush=True)

    print("\n[2/3] held-out (deck, boss) pairs: scripted floor vs "
          "generalist zero-shot vs per-deck RL(8k) ceiling", flush=True)
    hrng = random.Random(4242)
    schools = ("fire", "storm", "death", "ice", "life", "balance")
    rows = []
    for i, sc in enumerate(schools):
        boss = random_boss(hrng, f"held{i}")
        dl = sample_deck(legal_pool(cards, sc), sc, boss, hrng)
        sim = Sim(dict(cards), dl, sc, boss, player_hp=10**9,
                  rules=LIVE_RULES)
        heur = max((make_blade_stack(k) for k in (1, 2, 3)),
                   key=lambda p: evaluate(sim, p, n=400)[0])
        wh, mh = evaluate(sim, heur, n=2000)
        wg, mg = evaluate(sim, gen.policy(), n=2000)
        t1 = time.time()
        wf, mf, _ = fine_tune(cards, dl, sc, boss, LIVE_RULES, seed=i)
        tf = time.time() - t1
        rows.append(dict(school=sc, boss=boss.name, deck=sorted(set(dl)),
                         heuristic=[wh, mh], generalist=[wg, mg],
                         finetune=[wf, mf], finetune_s=round(tf, 1)))
        print(f"    {sc:<8} vs {boss.name:<18} "
              f"heur {wh*100:5.1f}%/{mh:5.2f}  "
              f"gen {wg*100:5.1f}%/{mg:5.2f}  "
              f"RL8k {wf*100:5.1f}%/{mf:5.2f} ({tf:.0f}s)", flush=True)
    gw = sum(r["generalist"][0] for r in rows) / len(rows)
    hw = sum(r["heuristic"][0] for r in rows) / len(rows)
    fw = sum(r["finetune"][0] for r in rows) / len(rows)
    print(f"    means: heuristic {hw*100:.1f}%  generalist {gw*100:.1f}%  "
          f"per-deck RL {fw*100:.1f}%", flush=True)

    print("\n[3/3] build_deck stage 2: per-deck fine-tune vs generalist "
          "zero-shot evaluation (same screen, same candidates)",
          flush=True)
    fresh = random_boss(random.Random(77), "fresh")
    for name, g in (("fine-tune", None), ("generalist", gen)):
        t1 = time.time()
        dl, w, m, _ = build_deck(cards, "death", fresh, LIVE_RULES,
                                 n_candidates=60, top_k=3, seed=7,
                                 generalist=g, log=None)
        print(f"    {name:<10} win {w*100:5.1f}%  ttk {m:5.2f}  "
              f"size {len(dl):>2}  ({time.time()-t1:.0f}s)", flush=True)

    json.dump({"held_out": rows},
              open("results_generalist.json", "w", encoding="utf-8"),
              indent=1)
    print("\nwrote generalist.json, results_generalist.json")
