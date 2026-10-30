"""Plot each configuration's change in pass^1 against the plain agent, with paired CIs.

Reads the per-task CSV written by `ras compare --csv` and writes a light and a dark
PNG for the README. Usage: .venv/bin/python scripts/plot_results.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from retail_agent.stats import paired_delta

CSV = Path("results/per_task.csv")
BASELINE = "main-test-vanilla"
# Top to bottom. Group headers are rows without a run.
ROWS = [
    ("Control", None),
    ("Plain agent, rerun", "main-test-vanilla-rep2"),
    ("Experiment 1: library mined from 74 train tasks", None),
    ("Past transcripts", "main-test-trajectories"),
    ("Injected skills", "main-test-skills_inject"),
    ("Skill index", "main-test-skills_index"),
    ("Experiment 2: libraries mined from 49 train tasks, skill index", None),
    ("Original miner", "exp2-test-all"),
    ("Original miner, gated", "exp2-test-all-gated"),
    ("Failure-only miner", "exp2-test-fail"),
    ("Failure-only miner, gated", "exp2-test-fail-gated"),
]
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", muted="#52514e", grid="#e4e3df", mark="#2a78d6"
    ),
    "dark": dict(surface="#1a1a19", ink="#f0efec", muted="#c3c2b7", grid="#3a3936", mark="#3987e5"),
}


def load() -> dict[str, dict[str, tuple[int, int]]]:
    runs: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    with CSV.open() as f:
        for r in csv.DictReader(f):
            runs[r["run"]][r["task_id"]] = (int(r["successes"]), int(r["trials"]))
    return runs


def plot(theme: str, runs) -> None:
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    n = len(ROWS)
    for i, (label, run) in enumerate(ROWS):
        y = n - 1 - i
        if run is None:
            ax.text(
                -0.215,
                y,
                label,
                color=c["ink"],
                fontsize=9.5,
                fontweight="bold",
                va="center",
                ha="left",
                transform=ax.get_yaxis_transform(),
            )
            continue
        d = paired_delta(runs[run], runs[BASELINE])
        lo, mid, hi = 100 * d.low, 100 * d.mean, 100 * d.high
        ax.plot([lo, hi], [y, y], color=c["mark"], linewidth=2, solid_capstyle="round")
        excludes_zero = hi < 0 or lo > 0
        ax.plot(
            mid,
            y,
            "o",
            markersize=7,
            markeredgewidth=2,
            markeredgecolor=c["mark"],
            markerfacecolor=c["mark"] if excludes_zero else c["surface"],
        )
        ax.text(
            -0.2,
            y,
            label,
            color=c["ink"],
            fontsize=9.5,
            va="center",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )
        value = f"{mid:+.1f}".replace("-", "\u2212")
        ax.text(9.8, y, value, color=c["muted"], fontsize=9, va="center", ha="right")
    ax.axvline(0, color=c["muted"], linewidth=1)
    ax.text(9.8, n - 1, "Δ", color=c["muted"], fontsize=9, va="center", ha="right")
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(-15, 10)
    ax.set_yticks([])
    ax.set_xlabel("Change in pass^1 vs. plain agent (percentage points)", color=c["muted"])
    ax.tick_params(axis="x", colors=c["muted"])
    ax.grid(axis="x", color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    fig.suptitle(
        "Mined skills never beat the plain agent on held-out tasks",
        x=0.02,
        ha="left",
        color=c["ink"],
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.905,
        "τ²-bench retail, 40 test tasks × 3 trials, DeepSeek-V4.1-Flash. "
        "Dot: mean over tasks; bar: 95% paired bootstrap CI; filled: CI excludes 0.",
        color=c["muted"],
        fontsize=8.5,
        ha="left",
    )
    fig.subplots_adjust(left=0.37, right=0.97, top=0.86, bottom=0.11)
    out = Path("docs") / f"results-{theme}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=c["surface"])
    print(f"wrote {out}")


if __name__ == "__main__":
    data = load()
    for t in THEMES:
        plot(t, data)
