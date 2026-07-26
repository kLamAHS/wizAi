"""
Exact solve (value iteration, stochastic shortest path) of the deck-free
abstraction — v2, rebuilt for non-stacking buffs.

Abstraction vs v1:
  - Buff state is no longer a count. It's a CONFIGURATION: one bit per
    distinct blade, and (charges+1) states per distinct ward (Fuel = 0..3
    charges pending). Same-name re-cast is an illegal action, as in-game.
  - Pips tracked as "effective pips for own school" 0..14. Off-school cards
    are charged 2 x pip cost (a pip slot is a power pip w.p. 0.85, worth 2
    same-school but only 1 off-school). Slight pessimism at low pip luck.
  - All cards always available (no hand/deck) => optimistic. Together with
    the pip pessimism this is still, empirically, a lower bound: draw
    randomness costs far more than odd-pip luck.

Objective: expected turns-to-kill with an immortal player (speed question).
"""
import numpy as np
from itertools import product
from w101_sim import load_cards, buff_applies

PMAX, PP, BUCKET = 14, 0.85, 25


def distinct_buffs(cards, decklist):
    blades, traps = [], []
    for n in dict.fromkeys(decklist):            # preserve order, dedupe
        c = cards[n]
        if c.kind == "blade":
            blades.append(c)
        elif c.kind == "trap":
            traps.append(c)
    return blades, traps


def solve(cards, decklist, boss, school, hp=None):
    boss_hp = hp if hp is not None else boss.hp
    H = int(np.ceil(boss_hp / BUCKET))
    blades, traps = distinct_buffs(cards, decklist)
    nukes = sorted({n for n in decklist if cards[n].kind == "damage"},
                   key=lambda n: cards[n].damage)
    nukes = [cards[n] for n in nukes]

    # buff configuration space: tuple(blade bits) + tuple(trap charge levels)
    blade_space = list(product(*[(0, 1)] * len(blades)))
    trap_space = list(product(*[range(c.charges + 1) for c in traps]))
    cfgs = [(b, t) for b in blade_space for t in trap_space]
    idx = {c: i for i, c in enumerate(cfgs)}
    NC = len(cfgs)

    def cost_eff(card):
        return card.pips if card.school == school else 2 * card.pips

    # per-config damage multiplier + post-hit config, per nuke
    def hit(cfg, nuke):
        b, t = cfg
        mult, nb = 1.0, list(b)
        for i, bl in enumerate(blades):
            if b[i] and buff_applies(bl, nuke.school):
                mult *= 1 + bl.percent
                nb[i] = 0
        nt = list(t)
        for i, tr in enumerate(traps):
            if t[i] and buff_applies(tr, nuke.school):
                mult *= 1 + tr.percent
                nt[i] = t[i] - 1
        return mult * boss.incoming_mult(nuke.school), idx[(tuple(nb), tuple(nt))]

    # actions: ("pass",), ("blade", i), ("trap", i), ("nuke", k)
    acts = [("pass", None)]
    acts += [("blade", i) for i in range(len(blades))]
    acts += [("trap", i) for i in range(len(traps))]
    acts += [("nuke", k) for k in range(len(nukes))]

    V = np.zeros((H + 1, PMAX + 1, NC))
    p_idx = np.arange(PMAX + 1)
    p2, p1 = np.minimum(p_idx + 2, PMAX), np.minimum(p_idx + 1, PMAX)
    h_idx = np.arange(H + 1)

    for _ in range(500):
        E = PP * V[:, p2, :] + (1 - PP) * V[:, p1, :]      # pip-gain expectation
        Q = np.full((len(acts), H + 1, PMAX + 1, NC), np.inf)
        for a, (kind, x) in enumerate(acts):
            if kind == "pass":
                Q[a] = 1 + E
                continue
            for ci, cfg in enumerate(cfgs):
                b, t = cfg
                if kind == "blade":
                    card = blades[x]
                    if b[x]:
                        continue                            # pending: illegal
                    c = cost_eff(card)
                    ni = idx[(b[:x] + (1,) + b[x + 1:], t)]
                    q = 1 + card.accuracy * E[:, :, ni] + (1 - card.accuracy) * E[:, :, ci]
                    Q[a, :, c:, ci] = q[:, c:]
                elif kind == "trap":
                    card = traps[x]
                    if t[x]:
                        continue
                    c = cost_eff(card)
                    ni = idx[(b, t[:x] + (card.charges,) + t[x + 1:])]
                    q = 1 + card.accuracy * E[:, :, ni] + (1 - card.accuracy) * E[:, :, ci]
                    Q[a, :, c:, ci] = q[:, c:]
                else:
                    card = nukes[x]
                    c = cost_eff(card)
                    mult, ni = hit(cfg, card)
                    dh = max(1, round(card.damage * mult / BUCKET))
                    h_new = np.maximum(h_idx - dh, 0)
                    succ = E[h_new][:, np.maximum(p_idx - c, 0), ni]
                    q = 1 + card.accuracy * succ + (1 - card.accuracy) * E[:, :, ci]
                    Q[a, :, c:, ci] = q[:, c:]
        Q[:, 0] = 0
        Vn = np.min(Q, axis=0)
        if np.max(np.abs(Vn - V)) < 1e-9:
            V = Vn
            break
        V = Vn
    pol = np.argmin(Q, axis=0)
    meta = dict(acts=acts, blades=blades, traps=traps, nukes=nukes,
                cfgs=cfgs, idx=idx, H=H)
    return V, pol, meta


def opening_line(V, pol, meta, max_len=12):
    """Greedy rollout of the abstract policy (successful casts, +2 pips/rd)."""
    h, p, cfg = meta["H"], 1, ((0,) * len(meta["blades"]), (0,) * len(meta["traps"]))
    line = []
    for _ in range(max_len):
        kind, x = meta["acts"][pol[h, p, meta["idx"][cfg]]]
        if kind == "pass":
            line.append("(pass)")
        elif kind == "blade":
            card = meta["blades"][x]
            line.append(card.name)
            b, t = cfg
            cfg = (b[:x] + (1,) + b[x + 1:], t)
            p = max(p - (card.pips if card.school else 0), 0)
        elif kind == "trap":
            card = meta["traps"][x]
            line.append(card.name)
            b, t = cfg
            cfg = (b, t[:x] + (card.charges,) + t[x + 1:])
        else:
            line.append(meta["nukes"][x].name)
            break
        p = min(p + 2, PMAX)
    return line


def dp_policy(V, pol, meta, school):
    """Transfer the abstract optimal policy into the full deck simulator,
    with the discard-to-dig mechanic when the wanted card isn't in hand."""
    acts, idx = meta["acts"], meta["idx"]
    bn = [c.name for c in meta["blades"]]
    tn = [c.name for c in meta["traps"]]

    def policy(sim, s):
        h = min(int(np.ceil(max(s.boss_hp, 0) / BUCKET)), meta["H"])
        p = min(s.norm_pips + 2 * s.pow_pips, PMAX)
        cfg = (tuple(1 if n in s.blades else 0 for n in bn),
               tuple(s.traps[n][1] if n in s.traps else 0 for n in tn))
        kind, x = acts[pol[h, p, idx[cfg]]]

        def dig(keep=None):
            """Discard one card so end-of-round draw can occur. Preference:
            pending-duplicate buffs > in-hand duplicates > weakest buff >
            weakest nuke (never the card we're digging for)."""
            if not s.deck or len(s.hand) < 7:
                return
            seen = set()
            def junk_rank(cd):
                pending_dup = (cd.kind == "blade" and cd.name in s.blades) or \
                              (cd.kind == "trap" and cd.name in s.traps)
                hand_dup = cd.name in seen
                seen.add(cd.name)
                if pending_dup: return (0, cd.percent)
                if hand_dup:    return (1, cd.percent if cd.kind != "damage" else cd.damage)
                if cd.kind != "damage": return (2, cd.percent)
                return (3, cd.damage)
            junk = [cd for cd in s.hand if cd.name != keep]
            if junk:
                s.hand.remove(min(junk, key=junk_rank))

        if kind == "pass":
            dig()
            return None
        want = {"blade": meta["blades"], "trap": meta["traps"],
                "nuke": meta["nukes"]}[kind][x]
        for card in s.hand:
            if card.name == want.name and sim.can_cast(s, card):
                return card
        if any(cd.name == want.name for cd in s.deck):
            dig(keep=want.name)
            return None
        # wanted card exhausted: substitute within kind, else best nuke
        subs = [cd for cd in s.hand if cd.kind == want.kind and sim.can_cast(s, cd)]
        if subs:
            key = (lambda cd: cd.damage) if want.kind == "damage" else (lambda cd: cd.percent)
            return max(subs, key=key)
        subs = [cd for cd in s.hand if cd.kind == "damage" and sim.can_cast(s, cd)]
        return max(subs, key=lambda cd: cd.damage, default=None)
    return policy


if __name__ == "__main__":
    from bosses import BOSSES, DECKS
    from w101_sim import Sim, Boss, evaluate, strat_nuke_asap, make_blade_stack
    cards = load_cards("cards_clean.json")
    boss = BOSSES["Jade Oni"]
    speed_boss = Boss(boss.name, boss.hp, boss.school, 0)   # immortal-player view
    for dname, deck in DECKS["fire"].items():
        V, pol, meta = solve(cards, deck, speed_boss, "fire")
        lb = V[meta["H"], 1, 0]
        sim = Sim(cards, deck, "fire", speed_boss, player_hp=10**9)
        w, m = evaluate(sim, dp_policy(V, pol, meta, "fire"), n=10000)
        wb, mb = max((evaluate(sim, make_blade_stack(k), n=4000) for k in range(6)),
                     key=lambda r: (r[0], -r[1]))
        print(f"{dname:<9} DP-LB {lb:6.2f} | transfer {m:6.2f} ({w*100:.0f}%) | "
              f"best heuristic {mb:6.2f} ({wb*100:.0f}%)")
        print("   opening:", " -> ".join(opening_line(V, pol, meta)))
