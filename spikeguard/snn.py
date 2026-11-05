"""Streaming spiking classifier: one time step per radar pulse, LIF neurons, spike-count readout.

Sized for an FPGA: 96-128-64-5 fully connected, so ~21k synapses that fit in block RAM as int8.
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

HIDDEN = (128, 64)


class SpikeNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=HIDDEN, beta=0.9):
        super().__init__()
        sizes = (n_in, *hidden, n_out)
        grad = surrogate.fast_sigmoid(slope=25)
        self.fcs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.lifs = nn.ModuleList(snn.Leaky(beta=beta, spike_grad=grad, learn_beta=True)
                                  for _ in self.fcs)
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sigma", torch.ones(n_in))

    def forward(self, x):
        """x: (batch, T, n_in) -> output spikes (T, batch, n_out), hidden spike counts per layer."""
        x = (x - self.mu) / self.sigma
        mems = [lif.init_leaky() for lif in self.lifs]
        out, counts = [], [0.0] * (len(self.fcs) - 1)
        for t in range(x.shape[1]):
            h = x[:, t]
            for i, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                h, mems[i] = lif(fc(h), mems[i])
                if i < len(counts):
                    counts[i] = counts[i] + h.sum()
            out.append(h)
        return torch.stack(out), counts


class MLP(nn.Module):
    """Non-spiking baseline of the same size, fed the time-averaged features."""

    def __init__(self, n_in, n_out, hidden=HIDDEN):
        super().__init__()
        sizes = (n_in, *hidden)
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(sizes[-1], n_out))
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sigma", torch.ones(n_in))

    def forward(self, x):
        return self.net(((x - self.mu) / self.sigma).mean(dim=1))


@torch.no_grad()
def quantize_(model, bits=8):
    """Symmetric per-layer post-training weight quantisation, in place."""
    q = 2 ** (bits - 1) - 1
    for fc in model.fcs:
        for p in (fc.weight, fc.bias):
            scale = p.abs().max() / q
            p.copy_(torch.round(p / scale) * scale)
