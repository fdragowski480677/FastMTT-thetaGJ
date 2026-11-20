"""Where does the resolution gain of the Gottfried-Jackson (GJ) component come from?

    python examples/gj_diagnostics.py --data-dir ../data_in --runs-dir ../results/gj \\
        --out-dir ../results/gj/diagnostics

Inputs, per channel: <data-dir>/<ch>_all.parquet and the batch_processing.py outputs
<runs-dir>/<ch>_nogj.parquet, <ch>_gauss.parquet, <ch>_rice.parquet (window on, other options default).

Outputs in --out-dir:
  resolution_3prong.png   pT resolution of 3-prong legs with a refitted SV, baseline vs GJ
  gain_vs_precision.png   change of the 68% half-width versus the angle precision sigma/theta_max,
                          with the shuffled-angle null test
  likelihood_shapes.png   gauss and rice likelihood versus x for a precise and a noisy measurement
  summary.md              the numbers behind the plots
  <ch>_<model>_null.npz   cached FastMTT leg pT of the null runs (delete to rerun)

Null test: the measured GJ angles, in units of theta_max, are shuffled between legs of similar
precision (sigma/theta_max) and FastMTT is rerun. The angle then says nothing about the leg it is
attached to, while the shape of the likelihood in x stays the same. Gain that survives the shuffle
does not come from the vertex measurement; it acts like a prior on x.
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "source"), HERE):
    if os.path.abspath(p) not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

import FastMTT  # noqa: E402
import Likelihood  # noqa: E402
from FastMTT_IO import process_FastMTT  # noqa: E402
from batch_processing import _parquet_df_to_fastmtt_inputs  # noqa: E402

CHANNELS = ("tt", "mt", "et")
MODELS = ("gauss", "rice")
PREC_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, np.inf)
N_SHUFFLE_BINS = 20
N_BOOT = 300

# Reference categorical palette, fixed order: baseline, gauss, rice
C_BASE, C_GAUSS, C_RICE = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLOR = {"gauss": C_GAUSS, "rice": C_RICE}


# ---------------------------------------------------------------------------------------------
# Per-leg quantities
# ---------------------------------------------------------------------------------------------
def leg_quantities(parsed):
    """theta_meas, sigma, theta_max and the active mask (N, 2), as FastMTT.run computes them."""
    f = FastMTT.FastMTT()
    lik = f.myLikelihood
    taus = parsed["measuredTauLeptons"]
    l1, l2 = taus[:, 0].copy(), taus[:, 1].copy()
    p1, p2 = f.get_p4(l1), f.get_p4(l2)
    l1, l2 = f.modify_lepton_mass(l1), f.modify_lepton_mass(l2)
    lik.mvisleg1, lik.mvisleg2 = l1[:, 4], l2[:, 4]
    lik.setLeptonInputs(p1, p2, l1[:, 0], l2[:, 0], l1[:, 5], l2[:, 5])
    v = parsed["vertex"]
    lik.setVertexInputs(v["PV"], v["SV"], v["covPV"], v["covSV"], v["hasSV"])
    return dict(active=lik.gj_active.copy(), theta=lik.gj_theta_meas.copy(),
                sigma=lik.gj_sigma.copy(), theta_max=lik.gj_theta_max.copy())


def shuffled_ratios(q, n_bins=N_SHUFFLE_BINS, seed=20261120):
    """theta_meas/theta_max of active legs, permuted within quantile bins of sigma/theta_max."""
    ratio = np.full(q["active"].shape, np.nan)
    idx = np.flatnonzero(q["active"].ravel())
    r = (q["theta"] / q["theta_max"]).ravel()[idx]
    prec = (q["sigma"] / q["theta_max"]).ravel()[idx]
    edges = np.quantile(prec, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, prec, side="right") - 1, 0, n_bins - 1)
    rng = np.random.default_rng(seed)
    out = np.empty_like(r)
    for b in range(n_bins):
        members = np.flatnonzero(which == b)
        out[members] = r[rng.permutation(members)]
    ratio.ravel()[idx] = out
    return ratio


# ---------------------------------------------------------------------------------------------
# Null runs: FastMTT with the measured angles replaced
# ---------------------------------------------------------------------------------------------
def install_theta_override():
    """Let FastMTT.run take theta_ratio (N, 2): the measured GJ angle in units of theta_max.

    Legs with NaN keep their measured angle. Runs without theta_ratio are unchanged.
    Installed in the parent process; forked workers inherit it.
    """
    if getattr(FastMTT.FastMTT.run, "_theta_override", False):
        return
    orig_run = FastMTT.FastMTT.run
    orig_set = Likelihood.Likelihood.setVertexInputs

    def run(self, *args, theta_ratio=None, **kwargs):
        self.myLikelihood._theta_ratio = theta_ratio
        return orig_run(self, *args, **kwargs)

    def setVertexInputs(self, *args, **kwargs):
        orig_set(self, *args, **kwargs)
        ratio = getattr(self, "_theta_ratio", None)
        if ratio is not None:
            use = self.gj_active & np.isfinite(ratio)
            self.gj_theta_meas = np.where(use, ratio*self.gj_theta_max, self.gj_theta_meas)

    run._theta_override = True
    FastMTT.FastMTT.run = run
    Likelihood.Likelihood.setVertexInputs = setVertexInputs


def null_run(parsed, model, ratio, workers, batch_size):
    vertex = dict(parsed["vertex"], theta_ratio=ratio)
    _m, _pt, pt1, pt2, _used = process_FastMTT(
        parsed["measuredTauLeptons"], parsed["measuredMETx"], parsed["measuredMETy"], parsed["covMET"],
        batch_size=batch_size, num_workers=workers, window=True, vertex=vertex, gj=True,
        gj_options={"gj_model": model}, return_gj_used=True)
    return np.stack([pt1, pt2], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------------------------
# Resolution metrics
# ---------------------------------------------------------------------------------------------
def w68(r, axis=None):
    q16, q84 = np.percentile(r, [16, 84], axis=axis)
    return 0.5*(q84 - q16)


def relative_change(r_model, r_base, rng, n_boot=N_BOOT):
    """100 * (w68_model / w68_base - 1) and its paired-bootstrap standard deviation."""
    value = 100*(w68(r_model)/w68(r_base) - 1)
    pick = rng.integers(0, r_base.size, size=(n_boot, r_base.size))
    boot = 100*(w68(r_model[pick], axis=1)/w68(r_base[pick], axis=1) - 1)
    return value, boot.std()


def leg_pt(path):
    d = pd.read_parquet(path, columns=["FastMTT_pt_1", "FastMTT_pt_2"])
    return np.stack([d["FastMTT_pt_1"], d["FastMTT_pt_2"]], axis=1).astype(np.float64)


def load_channel(ch, data_dir, runs_dir, out_dir, workers, batch_size, skip_null):
    df = pd.read_parquet(Path(data_dir) / f"{ch}_all.parquet")
    parsed = _parquet_df_to_fastmtt_inputs(df, channel=ch)
    q = leg_quantities(parsed)

    gen = np.stack([df["genPart_pt_1"], df["genPart_pt_2"]], axis=1).astype(np.float64)
    had = (np.abs(np.stack([df["genPart_pdgId_1"], df["genPart_pdgId_2"]], axis=1)) == 15) & (gen > 0)
    has_sv = np.stack([df["hasRefitSV_1"], df["hasRefitSV_2"]], axis=1).astype(bool)

    pt = {"baseline": leg_pt(Path(runs_dir) / f"{ch}_nogj.parquet")}
    for model in MODELS:
        pt[model] = leg_pt(Path(runs_dir) / f"{ch}_{model}.parquet")
    if not skip_null:
        ratio = shuffled_ratios(q)
        for model in MODELS:
            cache = Path(out_dir) / f"{ch}_{model}_null.npz"
            if cache.exists():
                pt[f"{model} null"] = np.load(cache)["pt"]
            else:
                pt[f"{model} null"] = null_run(parsed, model, ratio, workers, batch_size)
                np.savez_compressed(cache, pt=pt[f"{model} null"])
    res = {k: (v - gen)/np.where(gen > 0, gen, 1.0) for k, v in pt.items()}
    return dict(q=q, had=had, sv3=had & has_sv, gj=had & q["active"], res=res, n_events=len(df))


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_resolution(data, out):
    import matplotlib.pyplot as plt
    bins = np.arange(-0.6, 0.6 + 1e-9, 0.025)
    coarse = np.arange(-0.6, 0.6 + 1e-9, 0.05)    # difference panel: wider bins, less noise

    def density(r, edges):
        return np.histogram(r, bins=edges)[0]/(r.size*np.diff(edges))

    runs = [("baseline", "window (baseline)", C_BASE, "-"),
            ("gauss", "window + GJ gauss", C_GAUSS, "--"),
            ("rice", "window + GJ rice", C_RICE, ":")]
    chans = list(data)
    fig, axes = plt.subplots(2, len(chans), figsize=(4.6*len(chans), 6.4), facecolor=SURFACE,
                             gridspec_kw=dict(height_ratios=[3, 1.5]), sharex=True, squeeze=False)
    for j, ch in enumerate(chans):
        d = data[ch]
        sel = d["sv3"]
        top, bot = axes[0][j], axes[1][j]
        _style(top)
        _style(bot)
        diff_base = density(d["res"]["baseline"][sel], coarse)
        for i, (key, label, color, ls) in enumerate(runs):
            r = d["res"][key][sel]
            top.stairs(density(r, bins), bins, color=color, linestyle=ls, linewidth=2, label=label)
            top.plot([0.04, 0.12], [0.955 - 0.075*i]*2, transform=top.transAxes, color=color, linestyle=ls,
                     linewidth=2)
            top.text(0.14, 0.955 - 0.075*i, f"w68 = {w68(r):.3f}", transform=top.transAxes, va="center",
                     fontsize=9, color=INK2)
        top.text(0.97, 0.955, f"{sel.sum():,} legs", transform=top.transAxes, ha="right", va="center",
                 fontsize=9, color=INK2)
        top.set_title(f"{ch} channel", color=INK, fontsize=11)
        for key, _, color, ls in runs[1:]:
            bot.stairs(density(d["res"][key][sel], coarse) - diff_base, coarse, color=color, linestyle=ls,
                       linewidth=2)
        bot.axhline(0, color=C_BASE, linewidth=1.2)
        bot.set_xlabel(r"$(p_T^{reco} - p_T^{gen})\,/\,p_T^{gen}$", color=INK2)
    axes[0][0].set_ylabel("density", color=INK2)
    axes[1][0].set_ylabel("difference to\nbaseline", color=INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Hadronic 3-prong legs with a refitted SV: tau pT resolution", color=INK, fontsize=12.5,
                 y=0.995)
    fig.text(0.5, 0.01, "w68 = half-width of the central 68% interval (all legs, not only the plotted range).  "
             "Lower panels: density minus baseline density, in bins of 0.05.", ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.925))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def gain_table(data, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for ch, d in data.items():
        prec = d["q"]["sigma"]/d["q"]["theta_max"]
        for lo, hi in zip(PREC_EDGES[:-1], PREC_EDGES[1:]):
            sel = d["gj"] & (prec >= lo) & (prec < hi)
            base = d["res"]["baseline"][sel]
            row = dict(channel=ch, lo=lo, hi=hi, n=int(sel.sum()), w68_baseline=w68(base))
            for key in d["res"]:
                if key == "baseline":
                    continue
                row[key], row[f"{key} err"] = relative_change(d["res"][key][sel], base, rng)
            rows.append(row)
    return pd.DataFrame(rows)


def _bin_label(lo, hi):
    if lo == 0:
        return f"< {hi:g}"
    if np.isinf(hi):
        return f"> {lo:g}"
    return f"{lo:g} – {hi:g}"


def plot_gain(table, out):
    import matplotlib.pyplot as plt
    series = [(m + suffix, m, suffix == " null") for m in MODELS for suffix in ("", " null")]
    series = [s for s in series if s[0] in table]
    chans = list(dict.fromkeys(table["channel"]))
    fig, axes = plt.subplots(1, len(chans), figsize=(4.6*len(chans), 4.9), facecolor=SURFACE, sharey=True,
                             squeeze=False)
    offsets = np.linspace(-0.18, 0.18, len(series))
    for ax, ch in zip(axes[0], chans):
        _style(ax)
        t = table[table["channel"] == ch].reset_index(drop=True)
        xs = np.arange(len(t))
        ax.axhline(0, color=C_BASE, linewidth=1.5, label="window (baseline)")
        for off, (key, model, null) in zip(offsets, series):
            color = MODEL_COLOR[model]
            label = f"+ GJ {model}" + (", angles shuffled" if null else "")
            ax.errorbar(xs + off, t[key], yerr=t[f"{key} err"], color=color, linestyle="--" if null else "-",
                        linewidth=1.8, marker="o" if model == "gauss" else "s", markersize=7,
                        markerfacecolor=SURFACE if null else color, markeredgewidth=1.6, capsize=0,
                        elinewidth=1.2, label=label)
        ax.set_xticks(xs, [f"{_bin_label(lo, hi)}\n{n:,} legs" for lo, hi, n in zip(t["lo"], t["hi"], t["n"])],
                      fontsize=8.5)
        ax.set_xlabel(r"angle precision $\sigma_\theta\,/\,\theta_{max}$", color=INK2)
        ax.set_title(f"{ch} channel", color=INK, fontsize=11)
    axes[0][0].set_ylabel("change of 68% half-width vs baseline [%]\n(negative = better)", color=INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Hadronic legs entering the GJ term: resolution gain versus how well the angle is measured",
                 color=INK, fontsize=12.5, y=0.99)
    fig.text(0.5, 0.012, "Shuffled: each measured angle (in units of its theta_max) is swapped with one from another "
             "leg of similar precision, so it carries no information about the leg.  Error bars: paired bootstrap.",
             ha="center", fontsize=8.3, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.895))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def plot_likelihood_shapes(out, p_vis=40.0, m_vis=1.2):
    """Gauss and rice likelihood of one leg versus x, for a precise and a noise-dominated angle."""
    import matplotlib.pyplot as plt
    from scipy.special import i0e
    m_tau = FastMTT.FastMTT().myLikelihood.mTau
    E_vis = np.hypot(p_vis, m_vis)
    x = np.linspace(0.30, 0.999, 1400)
    theta = Likelihood.gj_angle_from_x(E_vis, p_vis, m_vis**2, x, m_tau)
    tmax = Likelihood.gj_angle_max(p_vis, m_vis**2, m_tau)
    allowed = x[theta > 0]
    cases = [("Precise angle", 0.2, 0.7), ("Noise-dominated angle", 1.5, 2.0)]   # sigma, theta_meas / theta_max

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), facecolor=SURFACE, sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1.25]))
    for j, (title, s, tm) in enumerate(cases):
        top, bot = axes[0][j], axes[1][j]
        _style(top)
        _style(bot)
        for ax in (top, bot):
            ax.axvspan(x[0], allowed[0], facecolor="none", edgecolor=GRID, hatch="///", linewidth=0)
        bot.text((x[0] + allowed[0])/2, 0.3, "kinematically\nforbidden", transform=bot.get_xaxis_transform(),
                 ha="center", va="center", fontsize=8.5, color=INK2,
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        top.fill_between(x, tm - s, tm + s, color=INK2, alpha=0.12, linewidth=0)
        top.axhline(tm, color=INK2, linewidth=1.2, linestyle="--")
        top.plot(x, theta/tmax, color=INK, linewidth=2)
        top.set_ylim(0, max(1.15, tm + s + 0.15))
        top.set_title(rf"{title}: $\sigma_\theta = {s:g}\,\theta_{{max}}$, "
                      rf"$\theta_{{meas}} = {tm:g}\,\theta_{{max}}$", color=INK, fontsize=10.5)

        t_m, sig = tm*tmax, s*tmax
        lg = np.exp(-0.5*((t_m - theta)/sig)**2)
        z = t_m*theta/sig**2
        lr = np.exp(-0.5*((t_m - theta)/sig)**2)*i0e(z)    # = exp(-(t_m^2 + theta^2)/2s^2) I0(z) up to a constant
        bot.plot(x, lg/lg.max(), color=C_GAUSS, linewidth=2)
        bot.plot(x, lr/lr.max(), color=C_RICE, linewidth=2, linestyle="--")
        bot.set_ylim(0, 1.08)
        bot.set_xlabel(r"$x = E_{vis} / E_\tau$", color=INK2)
    axes[0][0].set_ylabel(r"$\theta_{GJ}(x)\,/\,\theta_{max}$", color=INK2)
    axes[1][0].set_ylabel("GJ likelihood (max = 1)", color=INK2)
    axes[1][1].text(0.6, 0.3, "gauss: the same preference for mid-range x\nfor every such leg, whatever its true x\n"
                    "→ acts like a prior on x.   rice: ≈ flat.", transform=axes[1][1].transAxes, ha="center",
                    va="center", fontsize=9, color=INK2)
    handles = [Line2D([], [], color=INK, linewidth=2),
               (Patch(facecolor=INK2, alpha=0.12), Line2D([], [], color=INK2, linewidth=1.2, linestyle="--")),
               Line2D([], [], color=C_GAUSS, linewidth=2),
               Line2D([], [], color=C_RICE, linewidth=2, linestyle="--")]
    labels = [r"predicted $\theta_{GJ}(x)$", r"measured $\theta \pm \sigma_\theta$",
              "gauss (profile over azimuth)", "rice (integral over azimuth)"]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(rf"GJ likelihood of one leg versus x  ($|p_{{vis}}|$ = {p_vis:g} GeV, $m_{{vis}}$ = {m_vis:g} GeV)",
                 color=INK, fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def write_summary(data, table, out):
    lines = ["# GJ component diagnostics", "",
             "## 68% half-width of (pT_reco - pT_gen)/pT_gen", ""]
    keys = list(next(iter(data.values()))["res"])
    lines.append("| channel | legs | N | " + " | ".join(keys) + " |")
    lines.append("|---|---|---|" + "---|"*len(keys))
    for ch, d in data.items():
        for cat, name in (("sv3", "3-prong with SV"), ("gj", "entering the GJ term"), ("had", "all hadronic")):
            sel = d[cat]
            lines.append(f"| {ch} | {name} | {sel.sum()} | "
                         + " | ".join(f"{w68(d['res'][k][sel]):.4f}" for k in keys) + " |")
    lines += ["", "## Change of the 68% half-width vs baseline [%] by angle precision sigma/theta_max",
              "(legs entering the GJ term; +- paired-bootstrap standard deviation)", ""]
    cols = [c for c in table.columns if c not in ("channel", "lo", "hi", "n", "w68_baseline") and
            not c.endswith(" err")]
    lines.append("| channel | sigma/theta_max | N | w68 baseline | " + " | ".join(cols) + " |")
    lines.append("|---|---|---|---|" + "---|"*len(cols))
    for _, r in table.iterrows():
        lines.append(f"| {r['channel']} | {_bin_label(r['lo'], r['hi'])} | {r['n']} | {r['w68_baseline']:.4f} | "
                     + " | ".join(f"{r[c]:+.1f} ± {r[c + ' err']:.1f}" for c in cols) + " |")
    lines += ["", "## Angle precision of the legs entering the GJ term", ""]
    for ch, d in data.items():
        prec = (d["q"]["sigma"]/d["q"]["theta_max"])[d["gj"]]
        lines.append(f"- {ch}: median sigma/theta_max = {np.median(prec):.2f}, "
                     f"median theta_max = {1e3*np.median(d['q']['theta_max'][d['gj']]):.1f} mrad, "
                     f"median sigma = {1e3*np.median(d['q']['sigma'][d['gj']]):.1f} mrad, "
                     f"fraction with sigma/theta_max < 0.5: {np.mean(prec < 0.5):.2f}")
    Path(out).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=CHANNELS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--skip-null", action="store_true", help="skip the shuffled-angle null runs")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    mp.set_start_method("fork", force=True)   # workers must inherit the theta override
    install_theta_override()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {ch: load_channel(ch, args.data_dir, args.runs_dir, out, args.workers, args.batch_size, args.skip_null)
            for ch in args.channels}
    table = gain_table(data)
    plot_resolution(data, out/"resolution_3prong.png")
    plot_gain(table, out/"gain_vs_precision.png")
    plot_likelihood_shapes(out/"likelihood_shapes.png")
    write_summary(data, table, out/"summary.md")
    print((out/"summary.md").read_text())


if __name__ == "__main__":
    main()
