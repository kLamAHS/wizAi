"""
Decision-time search vs the two sequence bars.

Three learned-policy negatives isolated one missing capability:
wobble-free sequence-level planning. Before reaching for learned
sequence models, the honest first weapon is decision-time SEARCH —
determinized rollouts with a competent base policy literally plan
the rest of the fight, and unlike imitation (capped at its teacher)
search can EXCEED its base by deviating exactly where rollouts prove
it profitable on this draw.

Bar 1 — the healer fight (focus(bs2) = 46.4%, every learned policy
0%): search over target-aware candidates with with_focus(bs2) as the
rollout base.

Bar 2 — the balance X-pip fight (per-deck RL = 85.1%, generalist
53%): the held5 pair from results_generalist, regenerated from its
seeds. Search with the blade-stack base must discover the two-chunk
Judgement line by rollout, not by memory.

Search costs no training but heavy inference; n is set accordingly
and reported.
"""
import copy
import json
import random
import time

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from deck_builder import legal_pool, sample_deck, random_boss
from player_curves import base_power_pip
from search_policy import make_search_policy
from w101_sim import (Sim, Boss, evaluate_paired, make_blade_stack,
                      with_focus)

cards = load_spells_full()
bosses, _ = load_bosses_full()
results = {}

# ---- bar 1: the healer fight ----------------------------------------
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

print("[1/2] healer bar (focus(bs2) = 46.4%):", flush=True)
sim = Sim(dict(cards), ammo, "death", living, player_hp=10**9,
          power_pip=base_power_pip(20), rules=LIVE_RULES, enemies=[heal])
focus_base = with_focus(make_blade_stack(2))
t0 = time.time()
st = evaluate_paired(sim, {
    "focus(bs2)": focus_base,
    "search(k=8, focus-base)": make_search_policy(base=focus_base, k=8),
}, n=300)
for n_, s_ in st.items():
    m = s_["mean_ttk"]
    print(f"   {n_:<24} win {s_['win_rate']*100:5.1f}%"
          + (f"  mean {m:5.2f}" if m == m else ""), flush=True)
print(f"   (n=300, {time.time()-t0:.0f}s)", flush=True)
results["healer_bar"] = {n_: {k: v for k, v in s_.items()
                              if k != "ruleset"} for n_, s_ in st.items()}

# ---- bar 2: the balance X-pip fight ----------------------------------
# regenerate the exact held5 pair from results_generalist's seeds
hrng = random.Random(4242)
schools = ("fire", "storm", "death", "ice", "life", "balance")
pair = None
for i, sc in enumerate(schools):
    b = random_boss(hrng, f"held{i}")
    dl = sample_deck(legal_pool(cards, sc), sc, b, hrng)
    if i == 5:
        pair = (sc, dl, b)
assert pair[0] == "balance"
sc, dl, xboss = pair
print(f"\n[2/2] X-pip bar (per-deck RL = 85.1%) — {sc} vs "
      f"{xboss.name}, deck {sorted(set(dl))}:", flush=True)
xsim = Sim(dict(cards), dl, sc, xboss, player_hp=10**9,
           rules=LIVE_RULES)
t0 = time.time()
st = evaluate_paired(xsim, {
    "blade-stack(2)": make_blade_stack(2),
    "search(k=8)": make_search_policy(k=8),
    "search(k=16)": make_search_policy(k=16),
}, n=300)
for n_, s_ in st.items():
    m = s_["mean_ttk"]
    print(f"   {n_:<24} win {s_['win_rate']*100:5.1f}%"
          + (f"  mean {m:5.2f}" if m == m else ""), flush=True)
print(f"   (n=300, {time.time()-t0:.0f}s)", flush=True)
results["xpip_bar"] = {n_: {k: v for k, v in s_.items()
                            if k != "ruleset"} for n_, s_ in st.items()}


def _no_nan(o):
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, float) and o != o:
        return None
    return o


json.dump(_no_nan(results), open("results_sequence.json", "w",
                                 encoding="utf-8"), indent=1,
          allow_nan=False)
print("\nwrote results_sequence.json")
