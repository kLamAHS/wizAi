"""
Two questions the gear/pet layer makes answerable.

Q1 — WHERE DOES GEAR BUDGET ACTUALLY BECOME COMBAT POWER?

The report's headline claim is that progression "is less a smooth
curve than a series of benchmark jumps" at 30, 56/60, 90, 100, 110,
120. Testing THAT directly would be circular: the jumps are in the
tables, the tables are transcribed from the report, so of course
they reproduce. The non-circular question is whether a jump in stat
BUDGET converts into a proportional jump in combat, and the answer
need not be yes — a rating buys diminishing crit probability, and
damage percent compounds against a fixed boss differently than
resist does. So: rank the 5-level steps by stat-budget delta, rank
them by measured combat delta, and compare the two orderings.

Held fixed on purpose: the deck, the boss, the policy, the seeds.
Only gear moves. That isolates the marginal value of GEAR, which is
not the same as the marginal value of leveling — a real wizard also
gains spells, and this probe deliberately gives them none of that.

Q2 — TRIPLE-DOUBLE VS QUAD-CRITICAL PET

The two end-state builds the report identifies, on the same wizard.
Prediction stated before the run: quad-critical is worth EXACTLY
nothing in a criticals-off era (it is mechanically inert, not merely
weak), and should lose to triple-double against a high-block boss,
because block contests crit while nothing contests flat damage and
resist. Triple-double should be the uniformly safe pick and
quad-critical the conditional one.

Confidence: gear envelopes MODELED, pet talent ranges SOURCED — see
gear.py. The crit CURVE is modeled too (make_rating_crit), so Q2's
magnitudes depend on a modeled formula even though its DIRECTION
follows from the mechanic.
"""
import copy
import json
import time

from data_full import load_spells_full, LIVE_RULES
from gear import BREAKPOINTS, gear_stats, loadout, power_pip
from rulesets import CRIT_ERA
from w101_sim import Sim, Boss, evaluate_paired, make_blade_stack

cards = load_spells_full()
t0 = time.time()
K_CRIT = 250.0                      # make_rating_crit's default
DECK = ["Fireblade", "Fireblade", "Fire Trap", "Fire Trap", "Feint",
        "Fire Dragon", "Fire Dragon", "Sunbird", "Sunbird", "Fire Elf"]
POLS = {f"blade-stack({k})": make_blade_stack(k) for k in (1, 2, 3)}


def best_of(sim, n=400):
    st = evaluate_paired(sim, POLS, n=n)
    return max(st.items(), key=lambda kv: (round(kv[1]["win_rate"], 2),
                                           -kv[1]["mean_ttk"]))[1]


def stat_budget(level, rules=CRIT_ERA):
    """Expected damage multiplier from the stat sheet alone, using the
    engine's own crit model — the 'budget' side of the comparison."""
    st = gear_stats(level, "fire", rules=rules)
    dmg = 1 + sum(st.get("damage", {}).values())
    cr = st.get("crit", 0.0)
    p = min(0.85, cr / (cr + K_CRIT)) if cr else 0.0
    return dmg * (1 + p * (rules.crit_multiplier - 1)) * \
        (1 + power_pip(level))


# TTK against any ONE dummy is quantized by the scripted policy's
# blade-stack cycle: the wizard buffs for k turns and then nukes, so
# fights snap to cycle boundaries and a damage gain buys nothing until
# it crosses one. Two ways that bit this probe before it was fixed —
# a fixed 6000 HP dummy left the L120 arm PIP-bound (+0.00 turns from a
# +22% damage pet; 10000 gave +0.01, 16000 gave +1.57), and after
# scaling HP to the stat budget the criticals-off arm landed exactly on
# a boundary and returned 9.00 turns for all three pets. Neither is a
# fact about pets. So each arm is measured across a SPREAD of dummy
# sizes and the deltas averaged, which is what makes the numbers below
# stable rather than an artifact of where one dummy happened to sit.
HP_SPREAD = (0.7, 0.85, 1.0, 1.15, 1.3)


def sized_boss(name, level, rules=CRIT_ERA, block=0.0, scale=1.0,
               target_hp=13000):
    """A race dummy scaled to the wizard's stat budget."""
    hp = int(round(target_hp * scale * stat_budget(level, rules)
                   / stat_budget(100, CRIT_ERA)))
    b = Boss(name, hp, "death", 0)
    b.resist_map = {}
    b.boost_map = {}
    b.block_chance = block
    return b


def race_ttk(level, rules, block, pet):
    """Mean TTK across the dummy-size spread (see HP_SPREAD)."""
    lo = loadout(level, "fire", rules=rules, pet=pet, pet_age="mega")
    out = []
    for scale in HP_SPREAD:
        boss = sized_boss("d", level, rules, block, scale)
        sim = Sim(dict(cards), DECK, "fire", boss, player_hp=10**9,
                  rules=rules, player_stats=lo["player_stats"],
                  power_pip=lo["power_pip"])
        out.append(best_of(sim)["mean_ttk"])
    return sum(out) / len(out), out


# ---- Q1 ------------------------------------------------------------
# Measured as KILL CAPACITY — the largest boss HP the wizard beats at
# least half the time inside a fixed turn budget — rather than TTK
# against a fixed dummy. TTK cannot span this range: an 8.5x growth in
# stat budget leaves any single dummy unwinnable at L5 (10% win) or
# already dead at the pip floor by L120 (TTK pinned regardless of
# damage). Capacity has neither floor, is monotone, and is in the same
# units as the budget scalar it is compared against.
#
# Swept at THREE turn budgets, which is the load-bearing part. A single
# budget tells you where the steps are; three tell you whether the
# steps belong to the GEAR or to the turn budget itself.
TURN_BUDGETS = (6, 8, 10)
CAP_N = 800                 # a 0.5 threshold on a win rate is noisy: at
CAP_TOL = 0.015             # n=300 the step locations moved between runs
JUMP = 15.0                 # % capacity gain that counts as a felt step


def kill_capacity(level, turns, rules=CRIT_ERA, lo_hp=200, hi_hp=300000):
    """Bisect for the boss HP at which P(kill within `turns`) = 0.5."""
    lo = loadout(level, "fire", rules=rules)

    def beats(hp):
        b = Boss("cap", int(hp), "death", 0)
        b.resist_map = {}
        b.boost_map = {}
        sim = Sim(dict(cards), DECK, "fire", b, player_hp=10**9,
                  rules=rules, player_stats=lo["player_stats"],
                  power_pip=lo["power_pip"])
        st = evaluate_paired(sim, POLS, n=CAP_N, max_turns=turns)
        return max(v["win_rate"] for v in st.values()) >= 0.5

    if not beats(lo_hp):
        return float("nan")
    while hi_hp - lo_hp > max(25, CAP_TOL * lo_hp):
        mid = (lo_hp + hi_hp) / 2
        if beats(mid):
            lo_hp = mid
        else:
            hi_hp = mid
    return lo_hp


print("Q1: gear only — same deck, same policy, same seeds. Capacity = "
      "boss HP killable inside a turn budget.\n", flush=True)
LEVELS = list(range(5, 121, 5))
curve, jumps = {}, {}
for turns in TURN_BUDGETS:
    caps = {lvl: kill_capacity(lvl, turns) for lvl in LEVELS}
    steps = []
    for a, b in zip(LEVELS, LEVELS[1:]):
        ca, cb = caps[a], caps[b]
        ba, bb = stat_budget(a), stat_budget(b)
        if ca != ca or cb != cb:
            continue
        steps.append({"step": f"{a}->{b}", "to": b,
                      "capacity_gain_pct": 100 * (cb - ca) / ca,
                      "budget_gain_pct": 100 * (bb - ba) / ba})
    curve[turns] = {"capacity": caps, "steps": steps}
    felt = [s for s in steps if s["capacity_gain_pct"] >= JUMP]
    jumps[turns] = [s["step"] for s in felt]
    print(f"  {turns:2d}-turn budget — steps worth >={JUMP:.0f}% capacity: "
          + ", ".join(f"{s['step']} ({s['capacity_gain_pct']:.0f}%)"
                      for s in felt), flush=True)
    plateau = [s["capacity_gain_pct"] for s in steps
               if s["capacity_gain_pct"] < JUMP]
    print(f"     everything else: {len(plateau)} steps averaging "
          f"{sum(plateau)/max(len(plateau), 1):.1f}% — the plateaus",
          flush=True)

budget_steps = curve[TURN_BUDGETS[0]]["steps"]
by_budget = sorted(budget_steps, key=lambda s: -s["budget_gain_pct"])[:5]
print("\n  biggest STAT-BUDGET jumps: "
      + ", ".join(f"{s['step']} ({s['budget_gain_pct']:.0f}%)"
                  for s in by_budget), flush=True)

stable = set.intersection(*(set(v) for v in jumps.values()))
print(f"\n  felt jumps common to ALL THREE turn budgets: "
      f"{sorted(stable) or 'none'}", flush=True)
on_bp = [j for j in sorted(set().union(*(set(v) for v in jumps.values())))
         if int(j.split('->')[1]) in BREAKPOINTS
         or int(j.split('->')[1]) - 4 in BREAKPOINTS]
print(f"  felt jumps landing on a reported gear breakpoint: "
      f"{on_bp or 'none'}", flush=True)

# ---- Q2 ------------------------------------------------------------
# Measured in TTK, not win rate. A level-100 geared wizard beats every
# boss a single-target fire deck can reasonably be pointed at, so win
# rate sits at 100% for every arm and carries no signal at all — the
# same ceiling that made the first crit probe degenerate. The survival
# half is measured separately below, on a boss built to contest it.
PETS = [("no pet", None), ("triple-double", "triple-double"),
        ("quad-critical", "quad-critical")]
print("\nQ2a: pet value as a damage race, by era and by how much crit "
      "the GEAR already carries", flush=True)
print("(boss HP scales with the wizard's stat budget — see sized_boss "
      "for why a fixed dummy breaks this)\n", flush=True)
ARMS_Q2 = [("no crits (live)", LIVE_RULES, 100, 0.0),
           ("crit era", CRIT_ERA, 60, 0.0),
           ("crit era", CRIT_ERA, 100, 0.0),
           ("crit era", CRIT_ERA, 120, 0.0),
           ("crit era", CRIT_ERA, 100, 400.0)]
pets, deltas, ratios = {}, {}, {}
for ename, rules, lvl, block in ARMS_Q2:
    bname = "high-block" if block else "race-dummy"
    base_ttk, base_spread, arm = None, None, {}
    gear_crit = gear_stats(lvl, "fire", rules=rules).get("crit", 0.0)
    for label, arch in PETS:
        mean, spread = race_ttk(lvl, rules, block, arch)
        key = f"{ename}/L{lvl}/{bname}/{label}"
        pets[key] = {"mean_ttk": mean, "per_dummy_ttk": spread,
                     "gear_crit": gear_crit}
        if label == "no pet":
            base_ttk, base_spread = mean, spread
            print(f"  {ename:<15} L{lvl:<4} {bname:<11} "
                  f"gear crit {gear_crit:5.0f}   {label:<14} "
                  f"ttk {mean:5.2f}", flush=True)
        else:
            d = base_ttk - mean
            each = [b - p for b, p in zip(base_spread, spread)]
            half = (max(each) - min(each)) / 2      # spread across dummies
            deltas[key] = d
            arm[label] = d
            pets[key]["per_dummy_delta"] = each
            print(f"  {' ':<15} {' ':<5} {' ':<11} {' ':<16} "
                  f"{label:<14} ttk {mean:5.2f}  "
                  f"({d:+.2f} turns, +/-{half:.2f} across dummies)",
                  flush=True)
    ratios[f"{ename}/L{lvl}/{bname}"] = {
        "gear_crit": gear_crit,
        "quad_over_triple": (arm["quad-critical"] / arm["triple-double"]
                             if arm["triple-double"] else None)}

print("\n  the decision, as a ratio — crit pet value / damage pet value:",
      flush=True)
for k, v in ratios.items():
    r = v["quad_over_triple"]
    verdict = ("crit pet wins" if r and r > 1.15 else
               "damage pet wins" if r is not None and r < 0.85 else "tie")
    print(f"    {k:<34} gear crit {v['gear_crit']:5.0f}   "
          f"{'n/a' if r is None else format(r, '4.2f')}   {verdict}",
          flush=True)

print("\nQ2b: the survival half — a boss built to contest it, so the "
      "resist talents have something to do\n", flush=True)
killer = Boss("killer", 9000, "death", 1450)
killer.resist_map = {}
killer.boost_map = {}
surv = {}
for label, arch in PETS:
    lo = loadout(100, "fire", rules=CRIT_ERA, pet=arch, pet_age="mega")
    sim = Sim(dict(cards), DECK, "fire", copy.copy(killer),
              player_hp=lo["player_hp"], rules=CRIT_ERA,
              player_stats=lo["player_stats"], power_pip=lo["power_pip"])
    r = best_of(sim)
    surv[label] = {"win": r["win_rate"], "ttk": r["mean_ttk"],
                   "hp": lo["player_hp"]}
    print(f"  {label:<14} {lo['player_hp']:6d} HP   "
          f"win {r['win_rate']*100:5.1f}%   ttk {r['mean_ttk']:5.2f}",
          flush=True)
base = surv["no pet"]["win"]
for label, _a in PETS[1:]:
    d = (surv[label]["win"] - base) * 100
    deltas[f"survival/{label}"] = d
    print(f"    {label:<14} {d:+6.1f} win-rate points over no pet",
          flush=True)

json.dump({"setup": {"deck": DECK, "school": "fire",
                     "q1_metric": f"boss HP killable at >=50% inside a "
                                  f"turn budget (bisected, n={CAP_N})",
                     "q1_turn_budgets": list(TURN_BUDGETS),
                     "q2a_bosses": "race dummy scaled to the arm's stat "
                                   "budget (see sized_boss); high-block "
                                   "variant adds a 400 block rating",
                     "q2b_boss": "killer 9000hp, 1450 dmg/round, "
                                 "mortal wizard at gear HP",
                     "pet_age": "mega",
                     "confidence": "gear envelopes MODELED; pet talent "
                                   "ranges SOURCED; crit curve MODELED"},
           "q1_by_turn_budget": {str(k): v for k, v in curve.items()},
           "q1_felt_jumps": {str(k): v for k, v in jumps.items()},
           "q1_jumps_stable_across_budgets": sorted(stable),
           "q2a_pets": pets,
           "q2a_crit_pet_over_damage_pet": ratios,
           "q2b_survival": surv,
           "q2_pet_value": deltas},
          open("results_gear.json", "w", encoding="utf-8"), indent=1)
print(f"\nwrote results_gear.json ({time.time()-t0:.0f}s)", flush=True)
