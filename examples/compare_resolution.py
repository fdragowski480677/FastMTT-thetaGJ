"""Compare tau pT resolution of several FastMTT outputs (merged.parquet files).

    python examples/compare_resolution.py baseline=a.parquet "with GJ=b.parquet" --out-dir results/cmp

Resolution r = (pT_reco - pT_gen) / pT_gen, evaluated for hadronic legs only
(|genPart_pdgId| == 15), because for electron/muon legs genPart is the visible lepton.
Categories: all hadronic legs, 3-prong legs with a refitted SV (where the GJ component can act),
hadronic legs without SV, and the di-tau pT versus gen_boson_pT.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
STYLES = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1))]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

CATEGORIES = [
    ("had", "All hadronic legs"),
    ("sv", "3-prong legs with refitted SV"),
    ("nosv", "Hadronic legs without SV"),
    ("ditau", "Di-tau pT vs gen boson pT"),
]
COLUMNS = ["FastMTT_pt", "FastMTT_pt_1", "FastMTT_pt_2", "genPart_pt_1", "genPart_pt_2",
           "genPart_pdgId_1", "genPart_pdgId_2", "hasRefitSV_1", "hasRefitSV_2", "gen_boson_pT"]


def residuals(df):
    out = {k: [] for k, _ in CATEGORIES}
    for leg in (1, 2):
        gen = df[f"genPart_pt_{leg}"].to_numpy(float)
        ok = (np.abs(df[f"genPart_pdgId_{leg}"].to_numpy()) == 15) & (gen > 0)
        r = (df[f"FastMTT_pt_{leg}"].to_numpy(float) - gen) / np.where(gen > 0, gen, 1.0)
        sv = df[f"hasRefitSV_{leg}"].to_numpy(bool) if f"hasRefitSV_{leg}" in df else np.zeros(len(df), bool)
        out["had"].append(r[ok])
        out["sv"].append(r[ok & sv])
        out["nosv"].append(r[ok & ~sv])
    out = {k: np.concatenate(v) if v else np.empty(0) for k, v in out.items()}
    if "gen_boson_pT" in df:
        g = df["gen_boson_pT"].to_numpy(float)
        ok = g > 0
        out["ditau"] = (df["FastMTT_pt"].to_numpy(float)[ok] - g[ok]) / g[ok]
    return out


def metrics(r):
    r = r[np.isfinite(r)]
    if r.size == 0:
        return dict(N=0)
    q16, q50, q84 = np.percentile(r, [16, 50, 84])
    core = r[np.abs(r) < 1]
    return dict(N=r.size, median=q50, width68=0.5*(q84 - q16), core_mean=core.mean(), core_std=core.std(),
                f10=np.mean(np.abs(r) < 0.1), f20=np.mean(np.abs(r) < 0.2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=path/to/merged.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        label, path = spec.split("=", 1)
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(path).schema_arrow.names)
        df = pd.read_parquet(path, columns=[c for c in COLUMNS if c in available])
        runs.append((label, residuals(df)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, name in CATEGORIES:
        for label, res in runs:
            m = metrics(res.get(key, np.empty(0)))
            if m["N"]:
                rows.append(dict(category=name, run=label, **m))
    table = pd.DataFrame(rows)
    fmt = {"median": "{:+.3f}", "width68": "{:.3f}", "core_mean": "{:+.3f}", "core_std": "{:.3f}",
           "f10": "{:.3f}", "f20": "{:.3f}"}
    shown = table.copy()
    for c, f in fmt.items():
        shown[c] = shown[c].map(f.format)
    shown = shown.rename(columns={"width68": "68% half-width", "f10": "frac |r|<0.1", "f20": "frac |r|<0.2"})
    try:
        md = shown.to_markdown(index=False)
    except ImportError:
        md = shown.to_string(index=False)
    (out / "resolution_summary.md").write_text((f"# {args.title}\n\n" if args.title else "") + md + "\n")
    table.to_csv(out / "resolution_summary.csv", index=False)
    print(md)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.arange(-1.0, 1.0 + 1e-9, 0.04)
    present = [(k, n) for k, n in CATEGORIES if any(res.get(k, np.empty(0)).size for _, res in runs)]
    fig, axes = plt.subplots(1, len(present), figsize=(4.6*len(present), 4.2), facecolor=SURFACE, squeeze=False)
    for ax, (key, name) in zip(axes[0], present):
        ax.set_facecolor(SURFACE)
        for i, (label, res) in enumerate(runs):
            r = res.get(key, np.empty(0))
            if r.size == 0:
                continue
            m = metrics(r)
            ax.hist(np.clip(r, bins[0], bins[-1]), bins=bins, density=True, histtype="step", linewidth=2,
                    color=COLORS[i % len(COLORS)], linestyle=STYLES[i % len(STYLES)],
                    label=f"{label}  (w68 {m['width68']:.3f})")
        ax.set_title(name, color=INK, fontsize=11)
        ax.set_xlabel(r"$(p_T^{reco} - p_T^{gen}) / p_T^{gen}$", color=INK2)
        ax.tick_params(colors=INK2, labelsize=9)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    axes[0][0].set_ylabel("Density", color=INK2)
    if args.title:
        fig.suptitle(args.title, color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "resolution_comparison.png", dpi=150, facecolor=SURFACE)
    print(f"Saved {out / 'resolution_comparison.png'}")


if __name__ == "__main__":
    main()
