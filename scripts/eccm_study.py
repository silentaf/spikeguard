"""Monte Carlo ECCM study + range-Doppler gallery.

Outputs:
  figures/rd_gallery.png    range-Doppler maps per jammer, without / with the matched ECCM
  figures/pd_vs_jnr.png     detection probability and false targets vs jammer-to-noise ratio
  results/eccm_study.json   raw numbers
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spikeguard.config import RadarConfig, JAMMERS  # noqa: E402
from spikeguard import signals, processing as P, eccm  # noqa: E402

CFG = RadarConfig()
JAM_SUBBAND = 1          # the jammer set itself on the radar's default band
TARGET_SNR = -12.0       # dB per sample before ~42 dB of processing gain
JNRS = list(range(0, 35, 5))
TRIALS = 40
SENSE_PULSES = 16        # pulses used to recognise the jammer before adapting


def run_cpi(cfg, jam, jnr, use_eccm, targets, rng):
    state = dict(eccm.DEFAULT)
    if use_eccm:
        rx0, _ = signals.receive(cfg, SENSE_PULSES, state["mode"], state["subband"], targets,
                                 jam, jnr, rng, jam_subband=JAM_SUBBAND)
        state = eccm.select(cfg, jam, rx0, state)   # oracle label here; the SNN replaces it later
    rx, refs = signals.receive(cfg, cfg.n_pulses, state["mode"], state["subband"], targets,
                               jam, jnr, rng, jam_subband=JAM_SUBBAND)
    rd, dets = P.process_cpi(cfg, rx, refs, state["excise"])
    return rd, dets, state


def trial(args):
    jam, jnr, use_eccm, seed = args
    rng = np.random.default_rng(seed)
    targets = [(rng.uniform(450, 2600), rng.uniform(-0.8, 0.8) * CFG.vel_max, TARGET_SNR)]
    _, dets, _ = run_cpi(CFG, jam, jnr, use_eccm, targets, rng)
    found, false = P.score(CFG, dets, targets, CFG.n_pulses)
    return jam, jnr, use_eccm, found, false


def monte_carlo():
    jobs = [(j, jnr, e, 1000 * i + 7 * jnr + 3 * e + 17 * k)
            for k, j in enumerate(JAMMERS[1:]) for jnr in JNRS for e in (False, True)
            for i in range(TRIALS)]
    with Pool() as pool:
        rows = pool.map(trial, jobs, chunksize=8)
    out = {}
    for jam, jnr, e, found, false in rows:
        d = out.setdefault(jam, {}).setdefault("eccm" if e else "baseline", {}).setdefault(
            jnr, {"pd": 0.0, "false": 0.0})
        d["pd"] += found / TRIALS
        d["false"] += false / TRIALS
    return out


def plot_curves(res):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = dict(zip(JAMMERS[1:], ["#7f7f7f", "#d62728", "#1f77b4", "#9467bd"]))
    for jam, d in res.items():
        for key, ls in (("baseline", "--"), ("eccm", "-")):
            pd = [d[key][j]["pd"] for j in JNRS]
            fa = [d[key][j]["false"] for j in JNRS]
            label = f"{jam} ({'ECCM' if key == 'eccm' else 'no ECCM'})"
            axs[0].plot(JNRS, pd, ls, marker="o", color=colors[jam], label=label)
            axs[1].plot(JNRS, fa, ls, marker="o", color=colors[jam], label=label)
    axs[0].set(xlabel="Jammer-to-noise ratio per sample (dB)", ylabel="Probability of detection",
               title=f"True-target detection (SNR {TARGET_SNR:.0f} dB/sample, {TRIALS} trials/point)",
               ylim=(-0.03, 1.03))
    axs[1].set(xlabel="Jammer-to-noise ratio per sample (dB)", ylabel="False detections per CPI",
               title="False targets accepted by CFAR", yscale="symlog")
    for a in axs:
        a.grid(alpha=0.3)
    axs[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/pd_vs_jnr.png", dpi=150)


def describe(state):
    if state["subband"] != eccm.DEFAULT["subband"]:
        return f"hop to sub-band {state['subband']}"
    if state["excise"]:
        return "time-freq excision"
    if state["mode"] == "coded":
        return "per-pulse random codes"
    return "none applicable"


def plot_gallery():
    rng = np.random.default_rng(42)
    targets = [(900.0, 200.0, TARGET_SNR), (1800.0, -350.0, TARGET_SNR)]
    rng_m, vel = P.axes(CFG, CFG.n_pulses)
    keep = slice(CFG.pulse_samples, CFG.pri_samples - CFG.pulse_samples)
    fig, axs = plt.subplots(len(JAMMERS), 2, figsize=(11, 3.0 * len(JAMMERS)), sharex=True, sharey=True)
    for i, jam in enumerate(JAMMERS):
        for k, use in enumerate((False, True)):
            rd, dets, state = run_cpi(CFG, jam, 20, use, targets, rng)
            ax = axs[i, k]
            ax.imshow(10 * np.log10(rd[:, keep]), aspect="auto", origin="lower", vmin=0, vmax=45,
                      extent=[rng_m[keep][0], rng_m[keep][-1], vel[0], vel[-1]], cmap="viridis")
            for r, v, _ in targets:
                ax.plot(r, v, "o", mfc="none", mec="w", ms=12, mew=1.5)
            if dets:
                d = np.array(dets)
                ax.plot(rng_m[d[:, 1]], vel[d[:, 0]], "x", color="r", ms=5)
            found, false = P.score(CFG, dets, targets, CFG.n_pulses)
            tag = "no ECCM" if not use else "ECCM: " + describe(state)
            ax.set_title(f"{jam} @ JNR 20 dB, {tag}\n{found}/2 targets, {false} false",
                         fontsize=8)
            if k == 0:
                ax.set_ylabel("velocity (m/s)")
    for ax in axs[-1]:
        ax.set_xlabel("range (m)")
    fig.suptitle("Range-Doppler maps (dB over noise). o = true target, x = CFAR detection", y=1.0)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/rd_gallery.png", dpi=120)


if __name__ == "__main__":
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    print(f"range res {CFG.range_res:.1f} m, velocity res {CFG.vel_res:.1f} m/s, "
          f"unambiguous velocity +/-{CFG.vel_max:.0f} m/s")
    plot_gallery()
    res = monte_carlo()
    (ROOT / "results/eccm_study.json").write_text(json.dumps(res, indent=1))
    for jam, d in res.items():
        print(jam)
        for key in ("baseline", "eccm"):
            print(f"  {key:9s}", " ".join(f"{d[key][j]['pd']:.2f}/{d[key][j]['false']:.1f}" for j in JNRS))
    plot_curves(res)
