"""
Charts for the benchmark tables -> plots/*.png (embedded in the README).

Design rules (dataviz method): one axis per chart; categorical color
follows the POLICY entity in fixed slot order everywhere (heuristic blue,
DP-transfer orange, search aqua, RL yellow — a validated palette); text
and grid stay in neutral ink; thin marks with surface gaps; legends for
multi-series charts, selective direct labels only.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical palette (light mode), fixed slot order
C = {"heuristic": "#2a78d6", "dp-transfer": "#eb6834",
     "search": "#1baf7a", "RL": "#eda100", "generalist": "#8a5cc9"}
INK, INK2, GRID, BOUND = "#222222", "#555555", "#e5e5e5", "#444444"
OUT = Path("plots")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "figure.dpi": 150,
})


def _label(r):
    return f"{r['school']} vs {r['boss']}"


def live_ladder():
    """Grouped horizontal bars: win rate by policy per live matchup."""
    data = json.load(open("results_live.json", encoding="utf-8"))
    rows = data["matches"]
    pols = [("heuristic", "heuristic"), ("dp-transfer", "dp-transfer"),
            ("search(k=5)", "search"), ("RL", "RL")]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    h, gap = 0.19, 0.02
    for j, (key, slot) in enumerate(pols):
        ys, xs = [], []
        for i, r in enumerate(rows):
            win = r["rl"][0] if key == "RL" else \
                r["paired"][key]["win_rate"]
            ys.append(i + (j - 1.5) * (h + gap))
            xs.append(win * 100)
        ax.barh(ys, xs, height=h, color=C[slot],
                label={"search": "search(k=5)", "RL": "RL(20k)"}.get(
                    slot, slot))
        for y, x in zip(ys, xs):
            if x < 3:                       # direct-label only the zeros
                ax.text(1.5, y, "0%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{_label(r)} ({r['boss_hp']} HP)" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("win rate (%)")
    ax.set_xlim(0, 100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              frameon=False, fontsize=9)
    ax.set_title("Live data: baseline ladder by matchup",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "live_ladder.png", bbox_inches="tight")
    plt.close(fig)


def classic_gap():
    """Dot plot: DP lower bound vs realized RL TTK, classic table."""
    data = json.load(open("results.json", encoding="utf-8"))
    rows = data["speed_immortal"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, r in enumerate(rows):
        lb, (w, m) = r["dp_lb"], r["rl"]
        ax.plot([lb, m], [i, i], color=GRID, lw=2, zorder=1)
        ax.plot(lb, i, "o", color=BOUND, ms=7, zorder=2)
        ax.plot(m, i, "o", color=C["RL"], ms=8, zorder=3)
        if w < 0.999:
            ax.text(m + 0.25, i, f"{w*100:.0f}%", va="center",
                    fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(r) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("turns to kill (lower is better)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=BOUND, label="DP lower bound")
    ax.plot([], [], "o", color=C["RL"], label="RL realized (win% shown)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Classic ruleset: perfect-information bound vs learned "
                 "play", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "classic_gap.png")
    plt.close(fig)


def storm_curve():
    """RL learning curve on the matchup no scripted policy wins."""
    curve = json.load(open("rl_curve_storm.json", encoding="utf-8"))
    hyb = json.load(open("results_hybrid.json", encoding="utf-8"))["paired"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = [c["episode"] for c in curve]
    ys = [c["win"] * 100 for c in curve]
    ax.plot(xs, ys, color=C["RL"], lw=2)
    marks = {"search(heur-base)": hyb["search(heur-base)"]["win_rate"],
             "search(RL-base)": hyb["search(RL-base)"]["win_rate"]}
    for name, v in marks.items():
        ax.axhline(v * 100, color=C["search"], lw=1.4,
                   ls="--" if "heur" in name else "-", alpha=0.9)
        ax.text(xs[-1], v * 100 + 1.2, name, ha="right", fontsize=8.5,
                color=INK2)
    ax.text(xs[2], ys[2] + 4, "RL(20k) training", fontsize=8.5, color=INK2)
    ax.set_xlabel("training episodes")
    ax.set_ylabel("win rate (%)")
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.set_title("Storm vs Jade Oni: learning the X-pip line "
                 "(scripted heuristics: 0%)", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "storm_curve.png")
    plt.close(fig)


def survival_tradeoff():
    """Survival table: win rate vs speed trade-off per policy."""
    data = json.load(open("results.json", encoding="utf-8"))
    rows = data["survival"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    pols = [("heuristic", "heuristic"), ("survival_heuristic", "search"),
            ("rl", "RL")]
    names = {"heuristic": "plain heuristic",
             "search": "triage-wrapped", "RL": "RL(24k)"}
    h, gap = 0.24, 0.02
    for j, (key, slot) in enumerate(pols):
        for i, r in enumerate(rows):
            w = r[key][0] * 100
            y = i + (j - 1) * (h + gap)
            ax.barh(y, w, height=h, color=C[slot],
                    label=names[slot] if i == 0 else None)
            if w < 3:
                ax.text(1.5, y, "0%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['school']} vs {r['boss']}\n"
                        f"(HP {r['player_hp']} vs {r['boss_dmg']}/rd)"
                        for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("win rate (%)")
    ax.set_xlim(0, 100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Survival objective (classic): defense buys kill rate",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "survival.png")
    plt.close(fig)


def progression_chart():
    """Small multiples: the strategy ladder as spells unlock."""
    data = json.load(open("progression.json", encoding="utf-8"))
    rows = data["levels"]
    xs = [r["level"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.4), sharex=True)
    a0, a1, a2 = axes
    wins = [r["win"] * 100 for r in rows]
    a0.plot(xs, wins, color=C["RL"], lw=2, marker="o", ms=5)
    a0.set_ylabel("win rate (%)")
    a0.set_ylim(min(90, min(wins) - 3), 100.5)
    a1.plot(xs, [r["ttk"] for r in rows], color=C["heuristic"], lw=2,
            marker="o", ms=5)
    a1.set_ylabel("mean TTK")
    a2.plot(xs, [r["deck_size"] for r in rows], color=C["heuristic"], lw=2,
            marker="o", ms=5, label="deck size")
    a2.plot(xs, [r["capacity"] for r in rows], color=INK2, lw=1.4,
            ls="--", label="deck capacity")
    a2.plot(xs, [r["pool"] / 4 for r in rows], color=C["search"], lw=2,
            marker="o", ms=4, label="unlocked pool / 4")
    a2.set_ylabel("cards")
    a2.set_xlabel("wizard level")
    a2.legend(frameon=False, fontsize=8.5, loc="upper left")
    a0.set_title("Progression sweep (fire): the strategy ladder as "
                 "spells unlock", fontsize=11, loc="left")
    for a in axes:
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "progression.png")
    plt.close(fig)


def scorer_scatter():
    """Surrogate-predicted vs simulated screen win, held-out bosses.
    Bosses are encoded by marker shape (color stays reserved for the
    policy entity system-wide)."""
    data = json.load(open("scorer_results.json", encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 100], [0, 100], color=GRID, lw=1.2, zorder=1)
    ax.text(36, 28, "perfect", fontsize=8, color=INK2, ha="left",
            rotation=38, rotation_mode="anchor")
    marks = ["o", "s", "^", "D", "v", "P"]
    for i, t in enumerate(data["test"]):
        xs = [w * 100 for w in t["sim"]]
        ys = [p * 100 for p in t["pred"]]
        rho = ("n/a" if t["spearman"] is None
               else f"{t['spearman']:.2f}")
        ax.plot(xs, ys, marks[i % len(marks)], ms=5, mfc="none",
                mec=BOUND, mew=1.1, ls="", alpha=0.55,
                label=f"{t['boss']} [{t['school']}]  ρ={rho}")
    ax.set_xlabel("simulated screen win rate (%)")
    ax.set_ylabel("surrogate prediction (%)")
    ax.set_xlim(-4, 104)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.set_title("Deck scorer: ridge surrogate vs simulation "
                 "(held-out bosses)", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "scorer.png")
    plt.close(fig)


def generalist_ladder():
    """Held-out (deck, boss) pairs: scripted floor, zero-shot
    generalist, per-deck RL ceiling."""
    data = json.load(open("results_generalist.json", encoding="utf-8"))
    rows = data["held_out"]
    pols = [("heuristic", "heuristic"), ("generalist", "generalist"),
            ("finetune", "RL")]
    names = {"heuristic": "scripted heuristic",
             "generalist": "generalist (zero-shot)",
             "RL": "per-deck RL(8k)"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    h, gap = 0.24, 0.02
    for j, (key, slot) in enumerate(pols):
        ys = [i + (j - 1) * (h + gap) for i in range(len(rows))]
        xs = [r[key][0] * 100 for r in rows]
        ax.barh(ys, xs, height=h, color=C[slot], label=names[slot])
        for y, x in zip(ys, xs):
            if x < 3:
                ax.text(1.5, y, "0%", va="center", fontsize=8,
                        color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['school']} vs {r['boss']}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("win rate (%)")
    ax.set_xlim(0, 100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              frameon=False, fontsize=9)
    ax.set_title("Deck-conditioned generalist: one policy, any deck, "
                 "zero-shot (held-out pairs)", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "generalist.png", bbox_inches="tight")
    plt.close(fig)


def survival_builds():
    """Speed-built vs survival-built decks under fire, both regimes.
    Build objective is the entity here: gray = built immortal,
    green = built with death on the line (RL pilot for both)."""
    data = json.load(open("results_survival_build.json",
                          encoding="utf-8"))
    ms = list(data["matchups"].items())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cols = {"speed": "#9aa0a6", "survival": "#1baf7a"}
    h, gap = 0.3, 0.04
    for j, build in enumerate(("speed", "survival")):
        ys = [i + (j - 0.5) * (h + gap) for i in range(len(ms))]
        xs = [m["picks"][build]["RL(8k)"]["win_rate"] * 100
              for _, m in ms]
        ax.barh(ys, xs, height=h, color=cols[build],
                label=f"{build}-built deck")
        for y, x, (_, m) in zip(ys, xs, ms):
            dl = m["picks"][build]["deck"]
            ax.text(x + 1, y, f"{x:.0f}%", va="center", fontsize=8.5,
                    color=INK2)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([
        f"{lab}: {m['boss_dmg']}/rd"
        + (" +650 burst" if lab == "burst" else "")
        + f"\n({m['player_hp']} player HP)" for lab, m in ms],
        fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("win rate (%) — RL(8k) pilot trained under fire")
    ax.set_xlim(0, 106)
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              frameon=False, fontsize=9)
    ax.set_title("Build for the fight you're in (both damage regimes)",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "survival_builds.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    made = []
    for fn in (live_ladder, classic_gap, survival_tradeoff, storm_curve,
               progression_chart, scorer_scatter, generalist_ladder,
               survival_builds):
        try:
            fn()
            made.append(fn.__name__)
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")
    print("wrote:", ", ".join(made))
