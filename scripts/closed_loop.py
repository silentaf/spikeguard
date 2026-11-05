"""Closed loop: SNN recognises the jammer from 16 sensing pulses, ECCM adapts the next CPI.

Compares no ECCM / SNN-driven ECCM / oracle ECCM (true label), with randomised jammer parameters
and receiver gain error as in the training set.
Outputs figures/closed_loop.png and results/closed_loop.json.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spikeguard.config import RadarConfig, JAMMERS  # noqa: E402
from spikeguard import signals, processing as P, eccm  # noqa: E402
from spikeguard.dataset import T_STEPS  # noqa: E402
from spikeguard.features import pulse_features, N_FEATURES  # noqa: E402
from spikeguard.snn import SpikeNet  # noqa: E402

CFG = RadarConfig()
JAM_SUBBAND = 1
TARGET_SNR = -12.0
JNRS = list(range(0, 35, 5))
TRIALS = 40
POLICIES = ("none", "snn", "oracle")
_model = None


def _init():
    global _model
    torch.set_num_threads(1)
    _model = SpikeNet(N_FEATURES, len(JAMMERS))
    _model.load_state_dict(torch.load(ROOT / "results/snn.pt", map_location="cpu"))
    _model.eval()


@torch.no_grad()
def classify(rx, refs):
    x = torch.tensor(pulse_features(rx, refs))[None]
    spk, _ = _model(x)
    return JAMMERS[int(spk.sum(0).argmax())]


def trial(args):
    jam, jnr, seed = args
    rng = np.random.default_rng(seed)
    targets = [(rng.uniform(450, 2600), rng.uniform(-0.8, 0.8) * CFG.vel_max, TARGET_SNR)]
    common = dict(jam_subband=JAM_SUBBAND, jam_params=signals.random_jam_params(CFG, rng),
                  drfm_targets=signals.random_drfm_targets(CFG, rng, int(rng.integers(1, 6))))
    gain = 10 ** (rng.uniform(-2, 2) / 20)
    st0 = eccm.DEFAULT
    rx0, refs0 = signals.receive(CFG, T_STEPS, st0["mode"], st0["subband"], targets, jam, jnr, rng,
                                 **common)
    label = classify(rx0 * gain, refs0)
    out = {"label": label}
    for pol in POLICIES:
        st = st0 if pol == "none" else eccm.select(CFG, label if pol == "snn" else jam, rx0, st0)
        rx, refs = signals.receive(CFG, CFG.n_pulses, st["mode"], st["subband"], targets, jam, jnr,
                                   rng, **common)
        _, dets = P.process_cpi(CFG, rx, refs, st["excise"])
        out[pol] = P.score(CFG, dets, targets, CFG.n_pulses)
    return jam, jnr, out


def main():
    jobs = [(j, jnr, 7919 * i + 31 * jnr + 101 * k)
            for k, j in enumerate(JAMMERS) for jnr in JNRS for i in range(TRIALS)]
    with Pool(initializer=_init) as pool:
        rows = pool.map(trial, jobs, chunksize=4)
    res = {}
    for jam, jnr, out in rows:
        d = res.setdefault(jam, {}).setdefault(jnr, {"correct_label": 0.0, **{
            p: {"pd": 0.0, "false": 0.0} for p in POLICIES}})
        d["correct_label"] += (out["label"] == jam) / TRIALS
        for p in POLICIES:
            d[p]["pd"] += out[p][0] / TRIALS
            d[p]["false"] += out[p][1] / TRIALS
    (ROOT / "results/closed_loop.json").write_text(json.dumps(res, indent=1))

    for jam in JAMMERS:
        print(jam)
        print("  label acc", " ".join(f"{res[jam][j]['correct_label']:.2f}" for j in JNRS))
        for p in POLICIES:
            print(f"  {p:9s}", " ".join(f"{res[jam][j][p]['pd']:.2f}/{res[jam][j][p]['false']:.1f}"
                                       for j in JNRS))

    fig, axs = plt.subplots(2, len(JAMMERS), figsize=(18, 6.5), sharex=True, sharey="row")
    style = {"none": ("--", "#7f7f7f"), "snn": ("-", "#d62728"), "oracle": (":", "#1f77b4")}
    names = {"none": "no ECCM", "snn": "SNN-driven ECCM", "oracle": "oracle ECCM"}
    for k, jam in enumerate(JAMMERS):
        for p in POLICIES:
            ls, c = style[p]
            for row, key in enumerate(("pd", "false")):
                axs[row, k].plot(JNRS, [res[jam][j][p][key] for j in JNRS], ls, marker="o",
                                 color=c, label=names[p])
        acc = np.mean([res[jam][j]["correct_label"] for j in JNRS])
        axs[0, k].set(title=f"{jam}  (SNN label acc {acc:.0%})", ylim=(-0.03, 1.05))
        axs[1, k].set(xlabel="JNR (dB)", yscale="symlog")
        for row in range(2):
            axs[row, k].grid(alpha=0.3)
    axs[0, 0].set_ylabel("Probability of detection")
    axs[1, 0].set_ylabel("False detections per CPI")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Closed loop: SNN recognises the jammer in 16 pulses (320 us), radar adapts the next CPI")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/closed_loop.png", dpi=150)


if __name__ == "__main__":
    main()
