"""Generate the self-labelled jammer dataset, train the SNN (and an MLP baseline), evaluate.

Outputs:
  results/dataset.npz, results/snn.pt, results/snn_metrics.json
  figures/snn_confusion.png, figures/snn_accuracy.png
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spikeguard.config import JAMMERS  # noqa: E402
from spikeguard.dataset import generate, T_STEPS  # noqa: E402
from spikeguard.features import N_FEATURES  # noqa: E402
from spikeguard.snn import SpikeNet, MLP, quantize_  # noqa: E402

N_PER_CLASS = 2400
EPOCHS = 40
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def load_data():
    path = ROOT / "results/dataset.npz"
    if path.exists():
        d = np.load(path)
        return d["X"], d["y"], d["jnr"]
    X, y, jnr = generate(N_PER_CLASS)
    np.savez_compressed(path, X=X, y=y, jnr=jnr)
    return X, y, jnr


def split(n, rng):
    idx = rng.permutation(n)
    a, b = int(0.7 * n), int(0.85 * n)
    return idx[:a], idx[a:b], idx[b:]


def train(model, X, y, Xv, yv, loss_fn, predict):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best, best_state = 0.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X), device=DEV)
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]
            loss = loss_fn(model, X[b], y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        acc = (predict(model, Xv) == yv).float().mean().item()
        if acc >= best:
            best, best_state = acc, copy.deepcopy(model.state_dict())
        if ep % 5 == 4:
            print(f"  epoch {ep + 1:3d} loss {loss.item():.3f} val acc {acc:.3f}")
    model.load_state_dict(best_state)
    return model


def snn_loss(model, x, y):
    spk, counts = model(x)
    # cross-entropy on spike counts, plus a small penalty that keeps hidden activity sparse
    rate = sum(counts) / (x.shape[0] * x.shape[1] * sum(model.fcs[i].out_features for i in range(2)))
    return F.cross_entropy(spk.sum(0), y) + 0.5 * rate


@torch.no_grad()
def snn_predict(model, x, steps=None):
    model.eval()
    spk, _ = model(x)
    s = spk[:steps].sum(0) if steps else spk.sum(0)
    return s.argmax(1)


@torch.no_grad()
def spike_rate(model, x):
    _, counts = model(x)
    n = sum(model.fcs[i].out_features for i in range(2))
    return (sum(counts) / (x.shape[0] * x.shape[1] * n)).item()


def mlp_loss(model, x, y):
    return F.cross_entropy(model(x), y)


@torch.no_grad()
def mlp_predict(model, x):
    model.eval()
    return model(x).argmax(1)


def main():
    X, y, jnr = load_data()
    print(f"dataset {X.shape} on {DEV}")
    tr, va, te = split(len(X), np.random.default_rng(0))
    Xt = torch.tensor(X, device=DEV)
    yt = torch.tensor(y, device=DEV)
    mu = Xt[tr].mean(dim=(0, 1))
    sigma = Xt[tr].std(dim=(0, 1)) + 1e-3

    print("training SNN")
    snn_model = SpikeNet(N_FEATURES, len(JAMMERS)).to(DEV)
    snn_model.mu.copy_(mu)
    snn_model.sigma.copy_(sigma)
    snn_model = train(snn_model, Xt[tr], yt[tr], Xt[va], yt[va], snn_loss, snn_predict)

    print("training MLP baseline")
    mlp = MLP(N_FEATURES, len(JAMMERS)).to(DEV)
    mlp.mu.copy_(mu)
    mlp.sigma.copy_(sigma)
    mlp = train(mlp, Xt[tr], yt[tr], Xt[va], yt[va], mlp_loss, mlp_predict)

    Xte, yte = Xt[te], yt[te]
    pred = snn_predict(snn_model, Xte).cpu().numpy()
    y_np = y[te]
    acc = float((pred == y_np).mean())
    acc_mlp = float((mlp_predict(mlp, Xte) == yte).float().mean())
    rate = spike_rate(snn_model, Xte)
    acc_steps = [float((snn_predict(snn_model, Xte, s).cpu().numpy() == y_np).mean())
                 for s in range(1, T_STEPS + 1)]

    q_model = copy.deepcopy(snn_model)
    quantize_(q_model, 8)
    acc_q8 = float((snn_predict(q_model, Xte).cpu().numpy() == y_np).mean())

    bins = np.arange(-10, 31, 5)
    jnr_te = jnr[te]
    jammed = y_np != 0
    acc_jnr = [float((pred[m] == y_np[m]).mean()) for lo, hi in zip(bins[:-1], bins[1:])
               for m in [jammed & (jnr_te >= lo) & (jnr_te < hi)]]

    synapses = sum(fc.weight.numel() for fc in snn_model.fcs)
    metrics = dict(test_accuracy_snn=acc, test_accuracy_snn_int8=acc_q8, test_accuracy_mlp=acc_mlp,
                   hidden_spike_rate=rate, accuracy_vs_pulses=acc_steps,
                   jnr_bins=bins.tolist(), accuracy_vs_jnr=acc_jnr, synapses=synapses,
                   n_train=len(tr), n_test=len(te))
    (ROOT / "results/snn_metrics.json").write_text(json.dumps(metrics, indent=1))
    torch.save(snn_model.state_dict(), ROOT / "results/snn.pt")
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, list)}, indent=1))

    cm = np.zeros((len(JAMMERS),) * 2)
    for t, p in zip(y_np, pred):
        cm[t, p] += 1
    cm /= cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(JAMMERS)):
        for j in range(len(JAMMERS)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="w" if cm[i, j] > 0.5 else "k", fontsize=9)
    ax.set(xticks=range(len(JAMMERS)), yticks=range(len(JAMMERS)), xticklabels=JAMMERS,
           yticklabels=JAMMERS, xlabel="predicted", ylabel="true",
           title=f"SNN jammer recognition, test acc {acc:.1%}")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/snn_confusion.png", dpi=150)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(range(1, T_STEPS + 1), acc_steps, "o-")
    axs[0].axhline(acc_mlp, ls="--", color="gray", label=f"MLP baseline (all {T_STEPS} pulses)")
    axs[0].set(xlabel="pulses observed (1 pulse = 20 us)", ylabel="accuracy",
               title="Early decision: accuracy vs latency")
    axs[0].legend()
    centers = (bins[:-1] + bins[1:]) / 2
    axs[1].bar(centers, acc_jnr, width=4)
    axs[1].set(xlabel="JNR (dB)", ylabel="accuracy (jammed classes)", ylim=(0, 1.02),
               title="Recognition vs jammer power")
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/snn_accuracy.png", dpi=150)


if __name__ == "__main__":
    main()
