"""Render the README figures (light + dark SVG) from committed result files.

  python figures/make_figures.py

Every number drawn here is read from the result files in results/. Nothing is typed in by hand.
Colours: one blue and one orange series per theme, with light gridlines.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "panel": "#f3f2ee"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "s1": "#3987e5", "s2": "#d95926", "panel": "#242422"},
}
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path", "font.size": 11, "svg.hashsalt": "dacapo"})


RUN1 = "results/gpu_run1"
NO_META = {"Creator": None, "Date": None, "Format": None, "Type": None}


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())["results"]


def save(fig, path: Path, th: dict) -> None:
    fig.savefig(path, facecolor=th["surface"], metadata=NO_META)
    plt.close(fig)


def e1_pooled(shift: int, metric: str = "chroma.pc_acc") -> dict:
    """Mean over all rows (16 melodies x 80/120/160 BPM) per pathway, for the GPU run at the given shift."""
    import numpy as np

    rows = load(f"{RUN1}/e1_melody_paths_shift{shift}.json")["rows"]
    return {p: float(np.mean([r[metric] for r in rows if r["pathway"] == p])) for p in {r["pathway"] for r in rows}}


def style(ax, th, xmax=1.0):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], length=0, labelsize=10)
    ax.xaxis.grid(True, color=th["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlim(0, xmax)


def top_legend(fig, ax, th, y):
    """Legend as one row under the subtitle, so it never sits on the data."""
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.015, y), ncol=len(labels), frameon=False,
                     fontsize=9.5, handlelength=1.0, handletextpad=0.4, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(th["ink2"])


def hbar_groups(ax, th, labels, s1, s2, n1, n2, title, subtitle, note=None, chance=None, chance_label=None, legend_y=0.85,
                fmt="{:.2f}"):
    import numpy as np

    y = np.arange(len(labels))[::-1] * 1.0
    h = 0.34
    for vals, off, color, name in ((s1, h / 2 + 0.02, th["s1"], n1), (s2, -h / 2 - 0.02, th["s2"], n2)):
        ax.barh(y + off, vals, height=h, color=color, label=name, zorder=3)
        for yy, v in zip(y + off, vals):
            ax.text(v + 0.012, yy, fmt.format(v), va="center", ha="left", fontsize=9.5, color=th["ink2"], zorder=4)
    if chance is not None:
        ax.axvline(chance, color=th["muted"], linewidth=1, zorder=2)
        ax.text(chance + 0.008, y[0] + 0.62, chance_label, color=th["muted"], fontsize=9, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=th["ink"], fontsize=10.5)
    style(ax, th, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    fig = ax.figure
    top_legend(fig, ax, th, legend_y)
    fig.text(0.02, 0.965, title, color=th["ink"], fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.02, 0.905, subtitle, color=th["ink2"], fontsize=10.5, va="top")
    if note:
        fig.text(0.02, 0.025, note, color=th["muted"], fontsize=8.5, va="bottom")


def fig_e1(th, name):
    s3, s1 = e1_pooled(3), e1_pooled(1)
    paths = [("input", "Input itself (measurement ceiling)"), ("nofsq", "Dense 25 Hz latents as context"),
             ("vae_rt", "VAE round trip only"), ("codes_rt", "5 Hz plan tokens"),
             ("codes_reorch", "Plan tokens + new instrumentation"), ("nocodes", "No audio condition (chance)")]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=th["surface"])
    fig.subplots_adjust(left=0.33, right=0.97, top=0.76, bottom=0.14)
    hbar_groups(ax, th, [p[1] for p in paths], [s3[p[0]] for p in paths], [s1[p[0]] for p in paths],
                "shift 3.0 (recommended for turbo)", "shift 1.0 (library default)", "The plan channel carries little of the melody",
                "Melody pitch-class accuracy after each pathway through ACE-Step 1.5 (higher is better)",
                note=f"16 melodies x 80/120/160 BPM, one seed, NVIDIA L4, turbo 8 steps. Source: {RUN1}/e1_melody_paths_shift*.json")
    save(fig, OUT / f"e1_melody_paths_{name}.svg", th)


def fig_e1b(th, name):
    p = load(f"{RUN1}/e1b_probe.json")["probes"]["pitch_class"]
    cats = [("acc", "All frames"), ("acc_frames_steady", "Pitch steady within the frame"),
            ("acc_frames_with_note_change", "Note changes within the frame"), ("acc_unseen_timbre", "Unseen lead instrument")]
    fig, ax = plt.subplots(figsize=(8.6, 4.0), facecolor=th["surface"])
    fig.subplots_adjust(left=0.33, right=0.97, top=0.70, bottom=0.15)
    hbar_groups(ax, th, [c[1] for c in cats], [p["quant"][c[0]] for c in cats], [p["latent_pool5"][c[0]] for c in cats],
                "Plan tokens", "Audio latents at the same 5 Hz rate",
                "Most of the loss is in the tokenizer",
                "Linear probe: decode the melody's pitch class on held-out melodies",
                chance=p["majority_baseline"], chance_label="majority-class baseline", legend_y=0.835, fmt="{:.3f}",
                note=f"48 melodies x 3 lead instruments; 6 melodies held out for testing. Source: {RUN1}/e1b_probe.json")
    save(fig, OUT / f"e1b_probe_{name}.svg", th)


def fig_e0(th, name):
    import numpy as np

    a = load("results/cpu/e0_ruler.json")["rows_A"]
    rows = [("piano", "Piano lead"), ("saw_lead", "Synth lead"), ("choir_aahs", "Voice-like lead")]
    mean = lambda cond: float(np.mean([r["swift.RPA"] for r in a if r["cond"] == cond]))
    solo = [mean(f"{r}/solo") for r, _ in rows]
    band = [mean(f"{r}/acc90") for r, _ in rows]
    fig, ax = plt.subplots(figsize=(8.6, 3.4), facecolor=th["surface"])
    fig.subplots_adjust(left=0.22, right=0.97, top=0.66, bottom=0.18)
    y = np.arange(len(rows))[::-1] * 1.0
    for yy, s, b in zip(y, solo, band):
        ax.plot([b, s], [yy, yy], color=th["axis"], linewidth=2, zorder=2, solid_capstyle="round")
    ax.scatter(solo, y, s=90, color=th["s1"], edgecolors=th["surface"], linewidths=2, zorder=3, label="Played alone")
    ax.scatter(band, y, s=90, color=th["s2"], edgecolors=th["surface"], linewidths=2, zorder=3, label="Same melody, loud band")
    for yy, s, b in zip(y, solo, band):
        ax.text(s + 0.02, yy, f"{s:.2f}", va="center", fontsize=9.5, color=th["ink2"])
        ax.text(b - 0.02, yy, f"{b:.2f}", va="center", ha="right", fontsize=9.5, color=th["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], color=th["ink"], fontsize=10.5)
    style(ax, th, 1.08)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    top_legend(fig, ax, th, 0.80)
    fig.text(0.02, 0.965, "Pitch trackers fail when a band plays", color=th["ink"], fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.02, 0.885, "Pitch-tracker accuracy on a perfectly played melody (exact ground truth)", color=th["ink2"], fontsize=10.5, va="top")
    fig.text(0.02, 0.025, "9 melodies per condition. Source: results/cpu/e0_ruler.json",
             color=th["muted"], fontsize=8.5, va="bottom")
    save(fig, OUT / f"e0_ruler_{name}.svg", th)


def word_recall(words: list[str], hyp: str) -> float:
    """Fraction of the lyric line's words found in the transcript (multiset match, case-insensitive)."""
    import re
    from collections import Counter

    have, want = Counter(re.findall(r"[a-z']+", hyp.lower())), Counter(w.lower() for w in words)
    return sum(min(c, have[w]) for w, c in want.items()) / sum(want.values())


def fig_e4(th, name):
    import numpy as np

    rows = [r for r in load(f"{RUN1}/e4_sweep.json")["rows"] if r["split"] == "test"]
    words = {it["id"]: it["words"] for it in json.loads((ROOT / "experiments/e4_sing/manifest.json").read_text())["items"]}
    arms = sorted({r["arm"] for r in rows})
    pt = {a: (np.mean([r["npa"] for r in rows if r["arm"] == a]),
              np.mean([word_recall(words[r["id"]], r["hyp"]) for r in rows if r["arm"] == a])) for a in arms}
    both = sum(r["npa"] >= 0.5 and word_recall(words[r["id"]], r["hyp"]) >= 0.5 for r in rows)
    fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor=th["surface"])
    fig.subplots_adjust(left=0.10, right=0.97, top=0.76, bottom=0.17)
    ax.add_patch(plt.Rectangle((0.5, 0.5), 0.55, 0.55, facecolor=th["panel"], edgecolor=th["grid"], zorder=1))
    ax.text(0.77, 0.80, f"Both at least half right:\n{both} of {len(rows)} clips", ha="center", va="center",
            fontsize=9.5, color=th["muted"], zorder=2)
    for fam, color, marker, label in (("ctx_", th["s1"], "o", "Guide given as context, released after k of 8 steps"),
                                      ("sde_", th["s2"], "D", "Generation starts from the noised guide")):
        sel = [a for a in arms if a.startswith(fam)]
        ax.scatter([pt[a][0] for a in sel], [pt[a][1] for a in sel], s=64 if marker == "o" else 52, marker=marker,
                   color=color, edgecolors=th["surface"], linewidths=1.5, zorder=4, label=label)
    x0, y0 = pt["noguide"]
    ax.scatter([x0], [y0], s=80, color=th["ink"], edgecolors=th["surface"], linewidths=1.5, zorder=5)
    ax.text(x0 + 0.03, y0, "No guide", va="center", fontsize=10, color=th["ink"], zorder=5)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], length=0, labelsize=10)
    ax.grid(True, color=th["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Notes sung at the written pitch", color=th["ink2"], fontsize=10.5)
    ax.set_ylabel("Lyric words recognised", color=th["ink2"], fontsize=10.5)
    top_legend(fig, ax, th, 0.865)
    fig.text(0.02, 0.965, "Right notes cost the words", color=th["ink"], fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Each mark is one way of guiding the singer with the melody, sung or played without words",
             color=th["ink2"], fontsize=10.5, va="top")
    fig.text(0.02, 0.02, f"9 held-out melodies x 2 seeds per mark; words from a Whisper large-v3 transcript. Source: {RUN1}/e4_sweep.json",
             color=th["muted"], fontsize=8.5, va="bottom")
    save(fig, OUT / f"e4_tradeoff_{name}.svg", th)


RUN2 = "results/gpu_run2"


def fig_e5(th, name):
    """E5 v2: how well the gap matches each answer, per condition, as registered (ALL draws).

    One axis: each index is accuracy against an answer minus accuracy against a shuffled copy of it (0 = chance).
    Blue: the answer the pattern asks for. Orange: a plain copy of B. Ticks: the registered thresholds."""
    s = load(f"{RUN2}/e5v2_main.json")["summary"]
    ix, h4 = s["indices"]["ALL"], s["verdicts"]["H4"]
    rows = [("lit", "A B A _", [("rule", "CI_lit", 0.25)]),
            ("ctl", "A B C _", [("copy", "CI_ctl", None)]),
            ("rel+2", "A B A+2 _", [("rule", "RI+2", 0.15), ("copy", "LI+2", None)]),
            ("rel+5", "A B A+5 _", [("rule", "RI+5", 0.15), ("copy", "LI+5", None)]),
            ("rel−3", "A B A−3 _", [("rule", "RI-3", 0.15), ("copy", "LI-3", None)]),
            ("seq", "A A+2 A+4 _", []),
            ("ana", "A A+5 B _", [("rule", "ANA", 0.15)]),
            ("null", "silence _", [("copy", "CI_null", None)])]
    color = {"rule": th["s1"], "copy": th["s2"]}
    marker = {"rule": "o", "copy": "D"}
    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=th["surface"])
    fig.subplots_adjust(left=0.20, right=0.97, top=0.78, bottom=0.13)
    ax.axvline(0, color=th["axis"], linewidth=1, zorder=2)
    n = len(rows)
    for i, (cond, ctx, marks) in enumerate(rows):
        y = n - 1 - i
        offs = [0.0] if len(marks) == 1 else [0.17, -0.17]
        for (kind, key, thr), dy in zip(marks, offs):
            v = ix[key]
            ax.plot([v["lo"], v["hi"]], [y + dy, y + dy], color=color[kind], linewidth=2, solid_capstyle="round", zorder=3)
            ax.scatter([v["mean"]], [y + dy], s=64 if kind == "rule" else 52, marker=marker[kind], color=color[kind],
                       edgecolors=th["surface"], linewidths=2, zorder=4)
            ax.text(v["hi"] + 0.015, y + dy, f"{v['mean']:.2f}", va="center", fontsize=9.5, color=th["ink2"], zorder=4)
            if thr is not None:
                half = 0.12 if len(marks) == 2 else 0.2
                ax.plot([thr, thr], [y + dy - half, y + dy + half], color=th["ink2"], linewidth=1.5, zorder=3)
        if cond == "seq":
            k, m = h4["tests_ALL"]["binomial"]["k"], h4["tests_ALL"]["binomial"]["n"]
            ax.text(0.015, y, f"gap matched the next step A+6 in {k} of {m} draws (needed 20%)", va="center",
                    fontsize=9.5, color=th["ink2"], zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{c}   {ctx}" for c, ctx, _ in rows][::-1], color=th["ink"], fontsize=10.5)
    style(ax, th)
    ax.set_xlim(-0.08, 0.68)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 0.25, 0.5])
    ax.set_xticklabels(["0 (chance)", "0.25", "0.50"])
    ax.scatter([], [], s=64, marker="o", color=th["s1"], label="Gap vs the answer the pattern asks for")
    ax.scatter([], [], s=52, marker="D", color=th["s2"], label="Gap vs a plain copy of B")
    ax.plot([], [], color=th["ink2"], linewidth=1.5, marker="|", markersize=9, linestyle="none", label="Registered threshold")
    top_legend(fig, ax, th, 0.865)
    fig.text(0.02, 0.965, "The model repeats a pattern; transposing and extending it stay near zero", color=th["ink"],
             fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Match to an answer minus match to a shuffled answer, mean and 95% CI over 36 melodies",
             color=th["ink2"], fontsize=10.5, va="top")
    fig.text(0.02, 0.02, f"4 draws per melody and condition; flute lines, repaint, NVIDIA L4. Source: {RUN2}/e5v2_main.json",
             color=th["muted"], fontsize=8.5, va="bottom")
    save(fig, OUT / f"e5_motif_{name}.svg", th)


RUN3 = "results/gpu_run3"


def fig_b2(th, name):
    """B2 (Run B): share of the 24 test melodies whose kept draw (best of 4) is joint, per Stage T arm.

    Blue: q from Whisper large-v3, as registered. Orange: the same kept draws scored with the CTC recogniser's WR
    (the registered independent-ASR label). Merged arms are drawn once. Tick: the registered capability bar (P2)."""
    r = load(f"{RUN3}/b2_result.json")
    q_ctc = {c["run_id"]: c["q_ctc"] for c in r["clips"]}
    rows = [("S*", "Clock on the 7 lyric heads"),
            ("S*_b0", "No clock, same guide"),
            ("S*_rand", "Clock on 7 random heads"),
            ("S*_shuf", "Clock, words on the wrong notes"),
            ("ctx_oohs_k4", "No clock, stronger guide")]
    n_items = r["arms"]["S*"]["n_items"]
    fig, ax = plt.subplots(figsize=(8.6, 4.9), facecolor=th["surface"])
    fig.subplots_adjust(left=0.29, right=0.97, top=0.74, bottom=0.15)
    n = len(rows)
    for i, (arm, label) in enumerate(rows):
        y = n - 1 - i
        a = r["arms"][arm]
        wsp = sum(it["j"] for it in a["items"].values())
        ctc = sum(round(q_ctc[it["kept_run"]], 6) >= 0.5 for it in a["items"].values())
        ax.plot([ctc / n_items, wsp / n_items], [y, y], color=th["grid"], linewidth=2, zorder=2)
        ax.scatter([wsp / n_items], [y], s=64, marker="o", color=th["s1"], edgecolors=th["surface"], linewidths=2, zorder=4)
        ax.scatter([ctc / n_items], [y], s=52, marker="D", color=th["s2"], edgecolors=th["surface"], linewidths=2, zorder=4)
        ax.text(wsp / n_items + 0.025, y, f"{wsp}/{n_items}", va="center", fontsize=9.5, color=th["ink2"], zorder=4)
        ax.text(ctc / n_items - 0.025, y, f"{ctc}", va="center", ha="right", fontsize=9.5, color=th["ink2"], zorder=4)
    bar = r["rule"]["P2"]["needed"]
    ax.plot([bar, bar], [-0.5, n - 0.5], color=th["ink2"], linewidth=1.5, zorder=3)
    ax.text(bar + 0.01, n - 0.45, "capability bar", va="bottom", fontsize=9, color=th["ink2"])
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in rows][::-1], color=th["ink"], fontsize=10.5)
    style(ax, th)
    ax.set_xlim(-0.06, 1.0)
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.scatter([], [], s=64, marker="o", color=th["s1"], label="Words recognised by Whisper (registered)")
    ax.scatter([], [], s=52, marker="D", color=th["s2"], label="Words recognised by a second recogniser (CTC)")
    top_legend(fig, ax, th, 0.85)
    fig.text(0.02, 0.965, "The clocked recipe clears the bar by Whisper, not by a second recogniser",
             color=th["ink"], fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Share of 24 new melodies where the kept draw (best of 4) gets at least half the notes and half the words",
             color=th["ink2"], fontsize=10.5, va="top")
    fig.text(0.02, 0.02, f"Voice Oohs guide for the first 3 of 8 steps (4 for the stronger one); clock at beta 8. NVIDIA L4. Source: {RUN3}/b2_result.json",
             color=th["muted"], fontsize=8.5, va="bottom")
    save(fig, OUT / f"b2_clock_{name}.svg", th)


def hero(th, name):
    """Schematic of a planner-renderer song model, the measured melody survival on each path, and the hook."""
    s = e1_pooled(3)
    pct = lambda k: f"{round(s[k], 2) * 100:.0f}%"  # same rounding as the bar labels
    W, H = 960, 340
    ink, ink2, muted, panel, line = th["ink"], th["ink2"], th["muted"], th["panel"], th["axis"]

    def box(x, y, w, h, title, sub):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{panel}" stroke="{line}" stroke-width="1"/>'
                f'<text x="{x + w / 2}" y="{y + h / 2 - 4}" text-anchor="middle" font-size="17" font-weight="700" fill="{ink}">{title}</text>'
                f'<text x="{x + w / 2}" y="{y + h / 2 + 18}" text-anchor="middle" font-size="12.5" fill="{ink2}">{sub}</text>')

    def arrow(x1, y1, x2, y2, color, width=2.5):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2 - 9}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>'
                f'<path d="M{x2} {y2} l-11 -6 v12 z" fill="{color}"/>')

    parts = [f'<rect width="{W}" height="{H}" fill="{th["surface"]}"/>',
             f'<text x="30" y="40" font-size="20" font-weight="700" fill="{ink}">Where does the melody live in an AI song model?</text>',
             f'<text x="30" y="64" font-size="13.5" fill="{ink2}">Measured in ACE-Step 1.5 (open weights): how much of a known melody&#8217;s pitch survives each path</text>',
             box(30, 110, 170, 74, "Your intent", "melody · lyrics · style"),
             box(270, 110, 190, 74, "Planner", "language model"),
             box(640, 110, 290, 74, "Renderer", "flow-matching transformer"),
             arrow(200, 147, 270, 147, line),
             arrow(460, 147, 640, 147, th["s2"], 4),
             f'<text x="550" y="132" text-anchor="middle" font-size="12.5" fill="{ink2}">5 Hz plan tokens</text>',
             f'<text x="550" y="176" text-anchor="middle" font-size="15" font-weight="700" fill="{ink}">{pct("codes_rt")} of pitch classes</text>',
             f'<path d="M115 184 C 115 278, 520 278, 676 193" fill="none" stroke="{th["s1"]}" stroke-width="4" stroke-linecap="round"/>',
             f'<path d="M686 186 l-14 2 l7 11 z" fill="{th["s1"]}"/>',
             f'<text x="400" y="270" text-anchor="middle" font-size="12.5" fill="{ink2}">dense audio context (25 Hz latents)</text>',
             f'<text x="400" y="292" text-anchor="middle" font-size="15" font-weight="700" fill="{ink}">{pct("nofsq")} of pitch classes (input itself: {pct("input")})</text>',
             f'<line x1="830" y1="185" x2="830" y2="222" stroke="{th["s1"]}" stroke-width="1.5" stroke-dasharray="4 4"/>',
             f'<rect x="712" y="222" width="236" height="62" rx="12" fill="{th["surface"]}" stroke="{th["s1"]}" stroke-width="1.5"/>',
             f'<text x="830" y="248" text-anchor="middle" font-size="13" font-weight="700" fill="{ink}">Da Capo hooks every step</text>',
             f'<text x="830" y="269" text-anchor="middle" font-size="12" fill="{ink2}">read or change each step, no extra calls</text>',
             f'<text x="30" y="328" font-size="11" fill="{muted}">Pitch-class accuracy (chance about 10%), 16 melodies x 3 tempi, NVIDIA L4. Source: {RUN1}/e1_melody_paths_shift3.json</text>']
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">' + "".join(parts) + "</svg>")
    (OUT / f"hero_{name}.svg").write_text(svg)


if __name__ == "__main__":
    for name, th in THEMES.items():
        hero(th, name)
        fig_e1(th, name)
        fig_e1b(th, name)
        fig_e0(th, name)
        fig_e4(th, name)
        fig_e5(th, name)
        fig_b2(th, name)
    print(sorted(p.name for p in OUT.glob("*.svg")))
