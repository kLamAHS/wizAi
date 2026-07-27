"""
Treasure cards in the action space: fire vs Malistaire survival.

The roadmap claim under test — the classic table's hardest row, fire
(2900 HP, no school heals) vs Malistaire (6000 HP, 400 death/round),
is "winnable only through the sideboard". VERDICT: REFUTED. The TC
wiring works (reflex draw with room-making, same-round cast, agents
value TCs, tabular state carries TC names + own HP), but the fight is
infeasible with ANY of these sideboards, and the arithmetic says why:

  - the immortal DP lower bound for this deck is 11 turns; the player
    dies during round 8 — a 3+ round gap no cast order closes;
  - off-school TC heals are pip-catastrophic: Satyr costs 4 raw pips
    (~4 rounds of off-school income), so sustain-by-healing consumes
    the entire kill budget (six Satyrs: still 0%);
  - zero-pip Death Shields (-80%) trade 320 damage for one TEMPO turn
    each; surviving L rounds with S shield-turns needs
    400L - 320S <= 2900, capping S~6, L~12 — but 6000 HP through this
    deck needs ~7 damage-line turns ON TOP of the shields, and a
    5-cast line maxes out near 2800 damage. No solution exists.

What would flip it (all out of scope for a 1v1 cheatless sideboard):
resist/pierce gear, an ally to split aggro, or double the HP. The
arms below generate the 0% table that documents this honestly.
"""
import json
import time

from w101_sim import (Sim, load_cards, evaluate, evaluate_paired,
                      make_blade_stack, make_survival, with_tc_draw)
from bosses import BOSSES, DECKS, PLAYER_HP
from rl_agent import train_agent

cards = load_cards("cards_clean.json")
boss = BOSSES["Malistaire"]
school, dl = "fire", DECKS["fire"]["oneshot"]
hp = PLAYER_HP[school]
print(f"fire vs {boss.name} ({boss.hp} HP, {boss.dmg} {boss.school}/round)"
      f" at {hp} player HP — deck has no heals or shields", flush=True)

SIDEBOARDS = {
    "none": None,
    "defense": ["Satyr@tc"] * 4 + ["Death Shield@tc"] * 2,
    "shields": ["Death Shield@tc"] * 8,
    "offense": ["Helephant@tc", "Helephant@tc", "Fireblade@tc",
                "Fireblade@tc"],
}

report = {}
for name, side in SIDEBOARDS.items():
    print(f"\n== arm: {name}  sideboard={side or '[]'} ==", flush=True)
    sim = Sim(dict(cards), dl, school, boss, player_hp=hp, sideboard=side)
    scripted = max(
        (make_survival(make_blade_stack(k)) for k in (0, 2, 4)),
        key=lambda p: evaluate(sim, with_tc_draw(p), n=1500)[0])
    t0 = time.time()
    agent, rsim = train_agent(dict(cards), dl, school, boss,
                              episodes=24000, warm=True, seed=0,
                              player_hp=hp, sideboard=side)
    st = evaluate_paired(rsim, {
        "RL(24k)": agent.policy(),
        "triage+tc": with_tc_draw(scripted)}, n=4000)
    report[name] = dict(sideboard=side or [], **{
        pilot: {k: s[k] for k in ("win_rate", "mean_ttk", "p90_ttk",
                                  "mean_hp_left")}
        for pilot, s in st.items()})
    for pilot, s in st.items():
        print(f"   {name:<8} [{pilot:<9}] win {s['win_rate']*100:5.1f}%  "
              f"mean {s['mean_ttk']:5.2f}  p90 {s['p90_ttk']:3.0f}  "
              f"hp left {s['mean_hp_left']:5.0f}", flush=True)
    print(f"   ({time.time()-t0:.0f}s)", flush=True)

from w101_sim import Boss

drill = Boss("survival-drill", 2800, "death", 500)
print(f"\n#### control: fire vs {drill.name} ({drill.hp} HP, "
      f"{drill.dmg} death/round) at {hp} player HP — a MARGINAL fight, "
      "where sideboard value is decided by the pilot", flush=True)
control = {}
for name in ("none", "defense", "shields"):
    side = SIDEBOARDS[name]
    sim = Sim(dict(cards), dl, school, drill, player_hp=hp,
              sideboard=side)
    scripted = max(
        (make_survival(make_blade_stack(k)) for k in (0, 2, 4)),
        key=lambda p: evaluate(sim, with_tc_draw(p), n=1500)[0])
    agent, rsim = train_agent(dict(cards), dl, school, drill,
                              episodes=24000, warm=True, seed=0,
                              player_hp=hp, sideboard=side)
    st = evaluate_paired(rsim, {
        "RL(24k)": agent.policy(),
        "triage+tc": with_tc_draw(scripted)}, n=4000)
    control[name] = dict(sideboard=side or [], **{
        pilot: {k: s[k] for k in ("win_rate", "mean_ttk", "p90_ttk",
                                  "mean_hp_left")}
        for pilot, s in st.items()})
    for pilot, s in st.items():
        print(f"   {name:<8} [{pilot:<9}] win {s['win_rate']*100:5.1f}%  "
              f"mean {s['mean_ttk']:5.2f}  p90 {s['p90_ttk']:3.0f}  "
              f"hp left {s['mean_hp_left']:5.0f}", flush=True)

json.dump({"matchup": f"fire vs {boss.name}", "player_hp": hp,
           "boss_dmg": boss.dmg, "arms": report,
           "control": dict(boss_hp=drill.hp, boss_dmg=drill.dmg,
                           arms=control)},
          open("results_tc.json", "w", encoding="utf-8"), indent=1)
print("\nwrote results_tc.json")
