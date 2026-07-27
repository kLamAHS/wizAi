"""
Living bosses: the 2026 boss-AI report's three-layer model in play.

The report supports: configured spell pool + state-aware-but-imperfect
AI + deterministic scripts (our CheatRule layer already IS layer 3).
This experiment swaps the legacy flat-damage Krokopatra for a caster
with a storm pool and the hitter archetype, and asks where the SOLO-
FEASIBILITY FRONTIER sits under each boss model as a base-stat death
wizard levels from 12 to 20 (no gear, no wand — weaker than a real
player, per the report's gear-dominance caveat).

Per level, each boss model gets its OWN searched deck and RL
fine-tune (feasibility under best play against THAT model), evaluated
on paired seeds n=3000. Deck capacity is level-gated like the
progression sweep. Player HP is interpolated between the report's
L1/L120 anchors (the anchors are the data; the interpolation is the
documented approximation) and power pips follow the universal base
ramp.

Confidence: the pool/archetype/discipline knobs are 'modeled' — the
report documents WHAT bosses do, not the exact weights. Boss casts
route through the same charm/ward/crit/fizzle engine as player casts,
so player counterplay (mantles, dispels) bites the boss too.
"""
import copy
import json

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from deck_builder import build_deck, fine_tune
from player_curves import school_hp, base_power_pip
from w101_sim import (Sim, Boss, evaluate_paired, make_blade_stack,
                      make_survival)

cards = load_spells_full()
bosses, _ = load_bosses_full()

POOL = ["Thunder Snake", "Lightning Bats", "Storm Shark",
        "Stormblade", "Storm Shield"]

flat = bosses["Krokopatra"]
living = copy.copy(flat)
living.dmg = 0
living.pool = POOL
living.archetype = "hitter"
living.discipline = 0.7

print(f"solo-feasibility frontier vs Krokopatra ({flat.hp} HP, resist "
      f"{flat.resist_map}): flat {flat.dmg}/round vs living pool "
      f"{POOL}", flush=True)
print("(Krokopatra is 4-person dungeon content: a naked level-12 "
      "SHOULD lose solo — the question is at what level each boss "
      "model concedes the solo)", flush=True)

report = {}
for LEVEL in (12, 16, 20):
    HP = school_hp("death", LEVEL)
    PP = base_power_pip(LEVEL)
    CAP = min(10 + (LEVEL // 10) * 2, 16)
    print(f"\n#### level {LEVEL}: {HP} HP, {PP*100:.0f}% base power "
          f"pips, capacity {CAP}, level-gated deck", flush=True)
    row = {"player_hp": HP, "power_pip": PP}
    for name, boss in (("flat", flat), ("living", living)):
        dl, w0, m0, _ = build_deck(cards, "death", boss, LIVE_RULES,
                                   n_candidates=60, top_k=3, seed=12,
                                   level=LEVEL, player_hp=HP,
                                   power_pip=PP, capacity=CAP, log=None)
        w, m, agent = fine_tune(cards, dl, "death", boss, LIVE_RULES,
                                seed=12, player_hp=HP, power_pip=PP)
        sim = Sim(dict(cards), dl, "death", boss, player_hp=HP,
                  power_pip=PP, rules=LIVE_RULES)
        st = evaluate_paired(sim, {
            "RL(8k)": agent.policy(),
            "triage": make_survival(make_blade_stack(2))}, n=3000)
        row[name] = {"deck": dl, **{pilot: {k: s[k] for k in
                     ("win_rate", "mean_ttk", "p90_ttk",
                      "mean_hp_left")} for pilot, s in st.items()}}
        for pilot, s in st.items():
            print(f"   {name:<7} [{pilot:<7}] win "
                  f"{s['win_rate']*100:5.1f}%  mean {s['mean_ttk']:5.2f}"
                  f"  p90 {s['p90_ttk']:3.0f}  "
                  f"deck {sorted(set(dl))}", flush=True)
    report[LEVEL] = row

# role-segmentation demo at the feasible level: living boss + healer
# minion (the Loremaster/Omen pattern). Current policies cannot switch
# targets, so this measures the PRESSURE a healer adds, not the answer.
LEVEL = 20
HP, PP = school_hp("death", LEVEL), base_power_pip(LEVEL)
healer = Boss("krok-acolyte", 300, "life", 0,
              pool=["Sprite", "Pixie"], archetype="healer")
healer.resist_map = {}
healer.boost_map = {}
dl = report[20]["living"]["deck"]
w, m, agent = fine_tune(cards, dl, "death", living, LIVE_RULES,
                        seed=12, player_hp=HP, power_pip=PP,
                        enemies=[healer])       # train vs the real MDP
sim = Sim(dict(cards), dl, "death", living, player_hp=HP, power_pip=PP,
          rules=LIVE_RULES, enemies=[healer])
st = evaluate_paired(sim, {"RL(8k)": agent.policy()}, n=3000)["RL(8k)"]
healer_row = {k: st[k] for k in ("win_rate", "mean_ttk", "p90_ttk")}
print(f"\n   living+healer minion at 20: win {st['win_rate']*100:5.1f}%"
      f"  mean {st['mean_ttk']:5.2f}  p90 {st['p90_ttk']:3.0f}",
      flush=True)

# pilot ladder on the living boss: the stochastic opponent INVERTS
# the usual ranking. Tabular MC starves (sparse wins, wide visited-
# state distribution — enemy-state features and a scripted-advisor
# warm start were both tried and help only marginally), shallow
# determinized search inherits rollout noise, and the fixed scripted
# line is robust. Diagnosed, not assumed: the 2x2 (episodes x
# enemy-state features) and advisor runs are in the commit history.
from search_policy import make_search_policy

print("\n== pilot ladder on the living boss at 20 (same deck) ==",
      flush=True)
w24, m24, agent24 = fine_tune(cards, dl, "death", living, LIVE_RULES,
                              seed=12, player_hp=HP, power_pip=PP,
                              episodes=24000,
                              advisor=make_blade_stack(2))
lsim = Sim(dict(cards), dl, "death", living, player_hp=HP,
           power_pip=PP, rules=LIVE_RULES)
ladder = evaluate_paired(lsim, {
    "blade-stack(3)": make_blade_stack(3),
    "blade-stack(2)": make_blade_stack(2),
    "search(k=5)": make_search_policy(k=5),
    "RL(24k)+advisor": agent24.policy()}, n=1500)
for n, s in ladder.items():
    print(f"   {n:<16} win {s['win_rate']*100:5.1f}%  "
          f"mean {s['mean_ttk']:5.2f}", flush=True)
pilot_ladder = {n: {k: s[k] for k in ("win_rate", "mean_ttk",
                                      "p90_ttk")}
                for n, s in ladder.items()}

# one narrated fight so the boss's turn-by-turn behavior is auditable
sim = Sim(dict(cards), dl, "death", living, player_hp=HP, power_pip=PP,
          rules=LIVE_RULES, log_events=True)
import random
sim.rng = random.Random(4)
t, won, hp_left = sim.run(make_survival(make_blade_stack(2)))
trace = [e for e in sim.last_state.events
         if e["type"].startswith("enemy")]
print(f"\nsample living-boss fight at 20 "
      f"({'won' if won else 'lost'} in {t}):", flush=True)
for e in trace[:12]:
    print(f"   t{e['turn']:>2} {e['type']:<12} {e.get('card', '')}",
          flush=True)

def _no_nan(o):
    """Strict JSON: zero-win arms produce NaN stats; store null."""
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_no_nan(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


json.dump(_no_nan({"pool": POOL, "levels": report,
                   "healer_minion_at_20": healer_row,
                   "pilot_ladder_at_20": pilot_ladder,
                   "sample_trace": trace}),
          open("results_living.json", "w", encoding="utf-8"), indent=1,
          allow_nan=False)
print("\nwrote results_living.json")
