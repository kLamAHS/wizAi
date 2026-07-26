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
     "search": "#1baf7a", "RL": "#eda100"}
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


if __name__ == "__main__":
    made = []
    for fn in (live_ladder, classic_gap, survival_tradeoff, storm_curve):
        try:
            fn()
            made.append(fn.__name__)
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")
    print("wrote:", ", ".join(made))
