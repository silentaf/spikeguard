"""Countermeasure policy: map a recognised jammer class to a radar configuration for the next CPI."""
import numpy as np

from .config import RadarConfig

DEFAULT = {"mode": "fixed", "subband": 1, "excise": False}


def least_jammed_subband(cfg: RadarConfig, rx, current):
    """Passive spectrum sensing: pick the sub-band with the lowest received power."""
    P = np.mean(np.abs(np.fft.fft(rx, axis=1)) ** 2, axis=0)
    f = np.fft.fftfreq(rx.shape[1], 1 / cfg.fs)
    power = [P[np.abs(f - c) <= cfg.bandwidth / 2].mean() for c in cfg.subbands]
    power[current] = np.inf
    return int(np.argmin(power))


def select(cfg: RadarConfig, jam_class, rx, state=DEFAULT):
    s = dict(state)
    if jam_class == "spot":
        s["subband"] = least_jammed_subband(cfg, rx, s["subband"])
    elif jam_class == "sweep":
        s["excise"] = True
    elif jam_class == "drfm":
        s["mode"] = "coded"
    # barrage: no waveform trick helps; rely on processing gain (reported honestly as a limit)
    return s
