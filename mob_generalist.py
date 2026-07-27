"""
Learned target switching: the generalist grows a target dimension.

The scripted bar from mob_fights.py: focus(bs2) wins 46.4% of the
immortal ammo-deck fight vs living Krokopatra + healer acolyte, and
target-blind play wins 0%. Here the linear generalist gets (card,
target) actions with per-target features (overkill/kill-now vs THAT
target, a support flag, target HP) and trains on random fights where
a quarter of episodes include a living healer minion — nothing tells
it to kill the healer first; the only signal is the return.

1v1 behavior is bit-compatible: target expansion only exists in
multi-enemy states, card-only feature vectors are unchanged, and
pre-target policy files load with zero-padded weights.
"""
import copy
import json

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from generalist import train_generalist
from player_curves import school_hp, base_power_pip
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

mob = json.load(open("results_mob.json", encoding="utf-8"))
ammo = mob["ammo_deck"]
PP = base_power_pip(20)

print("[1/2] training the target-aware generalist: 45k episodes, "
      "25% with a living healer minion", flush=True)
gen = train_generalist(cards, episodes=45000, seed=0, rules=LIVE_RULES,
                       log=True, mob_frac=0.25)
gen.save("generalist_mob.json")

print("\n[2/2] zero-shot on the mob_fights bar (ammo deck, living "
      "Krokopatra + healer, immortal tempo view, n=2000):", flush=True)
sim = Sim(dict(cards), ammo, "death", living, player_hp=10**9,
          power_pip=PP, rules=LIVE_RULES, enemies=[heal])
st = evaluate_paired(sim, {
    "generalist": gen.policy(),
    "focus(bs2)": with_focus(make_blade_stack(2)),
    "blind(bs2)": make_blade_stack(2)}, n=2000)
for n, s in st.items():
    mean = s["mean_ttk"]
    print(f"   {n:<11} win {s['win_rate']*100:5.1f}%"
          + (f"  mean {mean:5.2f}" if mean == mean else ""), flush=True)

# does it actually open on the healer? audit one fight's first hit
import random
tsim = Sim(dict(cards), ammo, "death", living, player_hp=10**9,
           power_pip=PP, rules=LIVE_RULES, enemies=[heal],
           rng=random.Random(3), log_events=True)
pol = gen.policy()
s = tsim.new_state()
first_hit_target = None
for _ in range(12):
    a = pol(tsim, s)
    if isinstance(a, tuple) and a[0].kind in ("damage", "drain"):
        first_hit_target = tsim.last_state = None or a[1]
        break
    if a is not None:
        tsim.cast(s, a if not isinstance(a, tuple) else a[0],
                  target=a[1] if isinstance(a, tuple) else 0)
    tsim.end_round(s)
print(f"\n   first hit aimed at enemy index: {first_hit_target} "
      f"(1 = the healer)", flush=True)


def _no_nan(o):
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, float) and o != o:
        return None
    return o


json.dump(_no_nan({"bar": {n: {k: s[k] for k in
                               ("win_rate", "mean_ttk", "p90_ttk")}
                           for n, s in st.items()},
                   "first_hit_target": first_hit_target}),
          open("results_mob_generalist.json", "w", encoding="utf-8"),
          indent=1, allow_nan=False)
print("\nwrote generalist_mob.json, results_mob_generalist.json")
