"""
Auras vs Earthquake: the counterplay the repo owner asked for, measured.

THE ASK, verbatim: "Optional but I like star school auras for fights
with myth mobs since I hate losing blades to an earthquake."

Earthquake now works (it wipes every charm and ward on the enemy team,
damage first). Auras now exist as their own card type with their real
extracted values. This probe closes the loop by asking whether the
preference is actually right, on the fight it was stated about.

THREE THINGS IT SEPARATES, because they are easy to conflate:

  1. WHAT THE WIPE COSTS.  Same deck, same policy, myth enemy with and
     without Earthquake in its pool. Everything else held.
  2. WHAT THE AURA BUYS AGAINST IT.  Blades-only deck vs the same deck
     with Amplify, both against the Earthquake caster.
  3. WHETHER IT IS THE WIPE OR JUST THE DAMAGE.  An aura is +15%
     damage whether or not anything gets wiped, so arm 2 on its own
     cannot tell "survives the wipe" from "is a buff". The no-quake
     control answers it: if the aura's edge is the SAME with the wipe
     turned off, it bought damage, not insurance.

Storm at level 60 — Celestia, which is exactly where Amplify unlocks,
and the school the owner was running when they raised this.

    python aura_probe.py
"""
import copy
import json

from data_full import LIVE_RULES, load_bosses_full, load_spells_full
from deck_builder import legal_pool
from gear import loadout
from player_curves import base_power_pip
from w101_sim import Boss, Sim, evaluate_paired, make_blade_stack

LEVEL = 60
SCHOOL = "storm"
N = 4000

cards = load_spells_full()
bosses, _ = load_bosses_full()

gear = loadout(LEVEL, SCHOOL, None, LIVE_RULES, pet="triple-double")
HP = gear["player_hp"]
PP = max(gear.get("power_pip", 0.0), base_power_pip(LEVEL))

# A myth mob that actually holds the card in question. Built rather than
# scraped so the ONLY difference between the two enemy arms is whether
# Earthquake is in the pool — a real pair of bosses would differ in a
# dozen other ways and the comparison would prove nothing.
MYTH_POOL = ["Minotaur", "Humongofrog", "Mythblade", "Myth Trap"]
# Sized so the fight is CONTESTED, and the size matters more than it
# looks. At 3,400 HP a geared level-60 storm wizard wins 100% of the
# time with or without the wipe; at 9,000 it wins ~0% either way and the
# measured cost of the wipe comes back NEGATIVE, because the boss burns
# a 6-pip turn on a 370-damage spell instead of Minotaur and both arms
# are pinned on the floor. 7,000 puts the blades-only deck near 83%,
# which leaves the wipe somewhere to push. The sweep below reports all
# three rather than quietly picking the flattering one.
MOB_HP = 7000
HP_SWEEP = (5000, 7000, 9000)


def myth_mob(quake):
    b = Boss("Myth Mob", MOB_HP, "myth", 0,
             pool=MYTH_POOL + (["Earthquake"] if quake else []),
             archetype="hitter", discipline=0.7)
    b.resist_map = {}
    b.boost_map = {}
    b.start_pips = 2
    return b


pool = legal_pool(cards, SCHOOL, level=LEVEL)
HITS = ["Storm Lord", "Tempest", "Stormzilla"]
BLADES = ["Stormblade", "Stormblade", "Tri Blade", "Tri Blade",
          "Spirit Blade"]
BASE = [h for h in HITS if h in pool] * 2 + [b for b in BLADES if b in pool]
DECKS = {
    "blades only": BASE,
    "blades + Amplify": BASE + ["Amplify", "Amplify"],
    "blades + Fortify": BASE + ["Fortify", "Fortify"],
}

print(f"level-{LEVEL} {SCHOOL} wizard: {HP} HP, {PP*100:.0f}% power pips, "
      f"damage {gear['player_stats'].get('damage')}", flush=True)
print(f"myth mob: {MOB_HP} HP, pool {MYTH_POOL} (+ Earthquake in one arm)\n",
      flush=True)
for n, d in DECKS.items():
    print(f"  {n:<18} {len(d)} cards: {sorted(set(d))}", flush=True)

print(f"\n== what the wipe costs, across difficulty (blades only) ==",
      flush=True)
sweep = {}
for hp in HP_SWEEP:
    cell = {}
    for arm, quake in (("quake", True), ("no quake", False)):
        mob = myth_mob(quake)
        mob.hp = hp
        sim = Sim(dict(cards), DECKS["blades only"], SCHOOL, mob,
                  player_hp=HP, power_pip=PP, rules=LIVE_RULES,
                  player_stats=gear["player_stats"])
        st = evaluate_paired(sim, {"bs3": make_blade_stack(3)}, n=N)["bs3"]
        cell[arm] = {k: st[k] for k in ("win_rate", "mean_ttk")}
    sweep[hp] = cell
    d = (cell["no quake"]["win_rate"] - cell["quake"]["win_rate"]) * 100
    print(f"   mob {hp:>5} HP   with {cell['quake']['win_rate']*100:5.1f}%"
          f"   without {cell['no quake']['win_rate']*100:5.1f}%"
          f"   costs {d:+6.1f}"
          + ("   <- floor: both arms pinned, sign is meaningless"
             if max(cell['quake']['win_rate'],
                    cell['no quake']['win_rate']) < 0.20 else ""),
          flush=True)

rows = {}
print(f"\n== the decks, mob at {MOB_HP} HP ==", flush=True)
print(f"{'deck':<20}{'vs quake':>18}{'no quake':>18}{'wipe costs':>13}",
      flush=True)
for name, deck in DECKS.items():
    cell = {}
    for arm, quake in (("quake", True), ("no quake", False)):
        sim = Sim(dict(cards), deck, SCHOOL, myth_mob(quake), player_hp=HP,
                  power_pip=PP, rules=LIVE_RULES,
                  player_stats=gear["player_stats"])
        st = evaluate_paired(sim, {"bs3": make_blade_stack(3)}, n=N)["bs3"]
        cell[arm] = {k: st[k] for k in ("win_rate", "mean_ttk", "p90_ttk")}
    rows[name] = cell
    w_q, w_n = cell["quake"]["win_rate"], cell["no quake"]["win_rate"]
    print(f"{name:<20}{w_q*100:>10.1f}% / {cell['quake']['mean_ttk']:4.1f}"
          f"{w_n*100:>10.1f}% / {cell['no quake']['mean_ttk']:4.1f}"
          f"{(w_q - w_n)*100:>12.1f}", flush=True)

base_q = rows["blades only"]["quake"]["win_rate"]
base_n = rows["blades only"]["no quake"]["win_rate"]
print(f"\nEarthquake costs the blades-only deck "
      f"{(base_n - base_q)*100:.1f} points.", flush=True)
for name in ("blades + Amplify", "blades + Fortify"):
    eq = (rows[name]["quake"]["win_rate"] - base_q) * 100
    nq = (rows[name]["no quake"]["win_rate"] - base_n) * 100
    print(f"{name:<20} edge vs the quake caster {eq:+6.1f}, "
          f"edge with no quake {nq:+6.1f}  -> "
          + ("insurance: worth more where the board gets wiped"
             if eq > nq + 1.0 else
             "just a buff: the same edge either way"), flush=True)

json.dump({"level": LEVEL, "school": SCHOOL, "n": N, "player_hp": HP,
           "power_pip": PP, "myth_pool": MYTH_POOL, "mob_hp": MOB_HP,
           "decks": {k: v for k, v in DECKS.items()},
           "hp_sweep": {str(k): v for k, v in sweep.items()},
           "arms": rows},
          open("results_auras.json", "w", encoding="utf-8"), indent=1)
print("\nwrote results_auras.json")
