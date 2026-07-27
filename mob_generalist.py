"""
Learned target switching: exploration deficit or representation
deficit? A controlled separation.

The first attempt failed the scripted bar outright (0% vs 46.4%,
opening on the boss), with two credit fixes tried (raw MC, advantage
baseline). Two hypotheses survive:

  EXPLORATION — eps-greedy almost never completes a healer kill by
  chance, so the winning sequence barely exists in the data; the
  linear class could rank it if it ever saw it.
  REPRESENTATION — kill-the-healer is a multi-step commitment no
  memoryless linear ranking can express, data or no data.

Three arms decide, all evaluated on the mob_fights bar (ammo deck,
living Krokopatra + healer acolyte, immortal tempo, n=2000):

  self-play    — 45k episodes, eps-greedy only (the failed baseline)
  advisor      — same, but early exploration follows with_focus(bs2)
                 with decaying probability: completed healer kills
                 enter the on-policy data
  BC(focus)    — pure imitation: clone with_focus(bs2) demonstrations
                 (eps=0.1 noise, winners only) in the same linear
                 class — zero exploration burden

If either teacher arm clears ~40%, the class suffices and exploration
was the wall. If both stay near 0% WITH perfect demonstrations, the
representation deficit is proven and the sequence-model rung is
formally motivated.

VERDICT (four arms, all 0% vs the 46.4% bar): representation, and
sharper than expected. The on-bar demonstration cell quantified WHY:
the teacher itself collapses from 46.4% to 1% under 10% action noise
— across a ~25-decision grind the winning line tolerates almost no
deviation, so any policy that wobbles anywhere loses everywhere. A
memoryless linear ranking cannot be wobble-free across a sequence
this long; sequence-level consistency is the missing capability.
(Also a data lesson: noisy demonstrations of brittle lines destroy
the very wins they exist to demonstrate — future demo collection
needs eps near 0.02, not the offline default 0.1.)
"""
import copy
import json
import random

import numpy as np

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from deck_builder import legal_pool, sample_deck, random_boss
from generalist import (FEATS, GeneralistQ, legal_actions, phi,
                        train_generalist, _rand_support)
from offline_rl import train_bc
from player_curves import base_power_pip
from rl_agent import PASS, MAX_TURNS, FAIL_PENALTY
from w101_sim import (Sim, Boss, evaluate_paired, make_blade_stack,
                      with_focus)

cards = load_spells_full()
bosses, _ = load_bosses_full()

living = copy.copy(bosses["Krokopatra"])
living.dmg = 0
living.pool = ["Thunder Snake", "Lightning Bats", "Storm Shark",
               "Stormblade", "Storm Shield"]
living.archetype = "hitter"
living.discipline = 0.7
heal = Boss("krok-acolyte-1", 300, "life", 0, pool=["Sprite", "Pixie"],
            archetype="healer")
heal.resist_map = {}
heal.boost_map = {}

ammo = json.load(open("results_mob.json", encoding="utf-8"))["ammo_deck"]
PP = base_power_pip(20)
DEMO_A = 24                 # legal actions can exceed offline_rl's 10
                            # once targets multiply them


def _act_key(x):
    """Normalize an action for matching: focus wraps EVERY card in a
    (card, target) tuple, but legal_actions only targets foe-directed
    kinds — a teacher's bladed tuple must match the bare blade (the
    first demo run silently turned every teacher setup cast into a
    PASS this way, deleting the line it was meant to demonstrate)."""
    from generalist import FOE_KINDS
    if x is None or x == PASS:
        return ("__pass__", None)
    if isinstance(x, tuple):
        card, tgt = x
        return (card.name, tgt if card.kind in FOE_KINDS else None)
    return (x.name, None)


def _match_action(acts, a):
    want = _act_key(a)
    for i, act in enumerate(acts):
        if _act_key(act) == want:
            return i
    return 0                                    # PASS fallback


def collect_demos(n_eps=2500, mob_frac=0.5, seed=11, eps=0.1,
                  pair=None):
    """Focus-scripted demonstrations, logged in the offline_rl format
    (every legal action's features + the teacher's choice + return-
    to-go). `pair`=(school, deck, boss, extra) pins every episode to
    one fight — the on-distribution cell with no shift excuse."""
    rng = random.Random(seed)
    teacher = with_focus(make_blade_stack(2))
    schools = ("fire", "ice", "storm", "myth", "life", "death",
               "balance")
    pools = {sc: legal_pool(cards, sc) for sc in schools}
    P, M, C, G, W = [], [], [], [], []
    for ep in range(n_eps):
        if pair is not None:
            sc, dl, boss, extra = pair
            extra = [copy.copy(e) for e in extra]
        else:
            sc = schools[rng.randrange(len(schools))]
            boss = random_boss(rng, f"demo{ep}")
            dl = sample_deck(pools[sc], sc, boss, rng)
            if dl is None:
                continue
            extra = [_rand_support(rng)] if rng.random() < mob_frac \
                else []
        sim = Sim(dict(cards), dl, sc, boss, player_hp=10**9,
                  rules=LIVE_RULES, rng=random.Random(seed * 10**5 + ep),
                  enemies=extra)
        s = sim.new_state()
        steps, won = [], False
        while True:
            acts = legal_actions(sim, s)[:DEMO_A]
            a = teacher(sim, s)
            pick = _match_action(acts, a)
            if rng.random() < eps:
                pick = rng.randrange(len(acts))
            Pm = np.zeros((DEMO_A, len(FEATS)), np.float32)
            mask = np.zeros(DEMO_A, bool)
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
            if not any(e.alive for e in s.enemies):
                won = True
                break
            sim.end_round(s)
            if s.player_hp <= 0 or s.turn >= MAX_TURNS:
                break
        g = 0.0 if won else -FAIL_PENALTY
        for Pm, mask, pick in reversed(steps):
            g = -1.0 + g
            P.append(Pm)
            M.append(mask)
            C.append(pick)
            G.append(g / 10.0)
            W.append(won)
    return dict(phis=np.stack(P), mask=np.stack(M),
                chosen=np.array(C, np.int16),
                G=np.array(G, np.float32), won=np.array(W, bool))


def first_hit_target(gen):
    sim = Sim(dict(cards), ammo, "death", living, player_hp=10**9,
              power_pip=PP, rules=LIVE_RULES, enemies=[copy.copy(heal)],
              rng=random.Random(3))
    pol = gen.policy()
    s = sim.new_state()
    for _ in range(12):
        a = pol(sim, s)
        if isinstance(a, tuple) and a[0].kind in ("damage", "drain"):
            return a[1]
        if a is not None:
            if isinstance(a, tuple):
                sim.cast(s, a[0], target=a[1])
            else:
                sim.cast(s, a)
        sim.end_round(s)
    return None


print("[1/4] arm self-play: 45k episodes, eps-greedy only", flush=True)
arms = {"self-play": train_generalist(
    cards, episodes=45000, seed=0, rules=LIVE_RULES, mob_frac=0.25)}

print("[2/4] arm advisor: guided exploration follows focus(bs2) with "
      "decaying probability", flush=True)
arms["advisor"] = train_generalist(
    cards, episodes=45000, seed=0, rules=LIVE_RULES, mob_frac=0.25,
    advisor=with_focus(make_blade_stack(2)))

print("[3/4] arm BC(focus): clone 2500 focus-scripted demonstration "
      "episodes over the random distribution (winners only)",
      flush=True)
demos = collect_demos()
print(f"    {len(demos['G'])} decisions, "
      f"{demos['won'].mean()*100:.0f}% from wins", flush=True)
arms["BC(focus)"] = GeneralistQ(w=train_bc(demos, winners_only=True))

print("[4/4] arm BC(on-bar): clone the teacher ON THE BAR FIGHT "
      "itself — no distribution-shift excuse", flush=True)
demos_bar = collect_demos(pair=("death", ammo, living,
                                [copy.copy(heal)]), seed=77)
print(f"    {len(demos_bar['G'])} decisions, "
      f"{demos_bar['won'].mean()*100:.0f}% from wins", flush=True)
arms["BC(on-bar)"] = GeneralistQ(w=train_bc(demos_bar,
                                            winners_only=True))

print("\nzero-shot on the mob_fights bar (ammo deck, living Krok + "
      "healer, immortal, n=2000):", flush=True)
sim = Sim(dict(cards), ammo, "death", living, player_hp=10**9,
          power_pip=PP, rules=LIVE_RULES, enemies=[heal])
pilots = {n: g.policy() for n, g in arms.items()}
pilots["focus(bs2) bar"] = with_focus(make_blade_stack(2))
st = evaluate_paired(sim, pilots, n=2000)
opens = {n: first_hit_target(g) for n, g in arms.items()}
for n, s_ in st.items():
    mean = s_["mean_ttk"]
    extra = f"  opens on enemy {opens[n]}" if n in opens else ""
    print(f"   {n:<15} win {s_['win_rate']*100:5.1f}%"
          + (f"  mean {mean:5.2f}" if mean == mean else "") + extra,
          flush=True)


def _no_nan(o):
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, float) and o != o:
        return None
    return o


arms["advisor"].save("generalist_mob.json")
json.dump(_no_nan({
    "bar": {n: {k: s_[k] for k in ("win_rate", "mean_ttk", "p90_ttk")}
            for n, s_ in st.items()},
    "opening_targets": opens}),
    open("results_mob_generalist.json", "w", encoding="utf-8"),
    indent=1, allow_nan=False)
print("\nwrote generalist_mob.json, results_mob_generalist.json")
