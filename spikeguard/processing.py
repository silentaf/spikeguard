"""Receive chain: spectral excision, pulse compression, Doppler processing, CA-CFAR."""
import numpy as np
from scipy import ndimage

from .config import RadarConfig


def spectral_excision(rx, k=8.0, block=250):
    """Time-frequency excision: in short blocks (so a sweeping jammer stays narrowband), zero the
    frequency bins whose power exceeds k x the block median. ECCM against sweep/narrowband jammers."""
    n_p, n = rx.shape
    X = np.fft.fft(rx[:, :n // block * block].reshape(n_p, -1, block), axis=2)
    P = np.abs(X) ** 2
    X[P > k * np.median(P, axis=2, keepdims=True)] = 0
    out = rx.copy()
    out[:, :n // block * block] = np.fft.ifft(X, axis=2).reshape(n_p, -1)
    return out


def pulse_compress(rx, refs):
    """Matched filter each pulse with its own Hamming-weighted reference (low range sidelobes).
    Output noise power stays 1 per bin."""
    refs = refs * np.hamming(refs.shape[1])
    n = rx.shape[1]
    nfft = 1 << int(np.ceil(np.log2(n + refs.shape[1])))
    R = np.fft.fft(rx, nfft, axis=1)
    H = np.conj(np.fft.fft(refs, nfft, axis=1))
    energy = np.sum(np.abs(refs) ** 2, axis=1, keepdims=True)
    return np.fft.ifft(R * H, axis=1)[:, :n] / np.sqrt(energy)


def range_doppler(mf):
    """Hann-windowed slow-time FFT -> power map (doppler_bins, range_bins), zero Doppler centred."""
    w = np.hanning(mf.shape[0])[:, None]
    w = w / np.sqrt(np.mean(w**2))
    rd = np.fft.fftshift(np.fft.fft(mf * w, axis=0), axes=0) / np.sqrt(mf.shape[0])
    return np.abs(rd) ** 2


def ca_cfar(rd, pfa=1e-6, guard=(1, 3), train=(4, 12)):
    """2-D cell-averaging CFAR plus 3x3 local-max clustering. Returns list of (doppler_idx, range_idx)."""
    gd, gr = guard
    td, tr = train
    outer = np.ones((2 * (gd + td) + 1, 2 * (gr + tr) + 1))
    outer[td:td + 2 * gd + 1, tr:tr + 2 * gr + 1] = 0
    n_train = outer.sum()
    noise = ndimage.convolve(rd, outer, mode="wrap") / n_train
    alpha = n_train * (pfa ** (-1 / n_train) - 1)
    hit = (rd > alpha * noise) & (rd == ndimage.maximum_filter(rd, size=3, mode="wrap"))
    return list(zip(*np.nonzero(hit)))


def process_cpi(cfg: RadarConfig, rx, refs, excise=False):
    if excise:
        rx = spectral_excision(rx)
    rd = range_doppler(pulse_compress(rx, refs))
    # Blank the eclipsed zone (receiver off while transmitting) and the last pulse-length,
    # where the matched filter only partially overlaps the window.
    floor = np.median(rd)
    rd[:, :cfg.pulse_samples] = floor
    rd[:, cfg.pri_samples - cfg.pulse_samples:] = floor
    return rd, ca_cfar(rd, cfg.pfa)


def target_cell(cfg: RadarConfig, r, v, n_pulses):
    fd = cfg.vel_to_doppler(v)
    d_idx = int(round(fd * cfg.pri * n_pulses)) % n_pulses
    d_idx = (d_idx + n_pulses // 2) % n_pulses  # fftshift
    return d_idx, cfg.range_to_delay(r)


def score(cfg, dets, targets, n_pulses, tol=(1, 2)):
    """Return (n_true_targets_detected, n_false_detections)."""
    cells = [target_cell(cfg, r, v, n_pulses) for r, v, _ in targets]
    found = set()
    false = 0
    for d, r in dets:
        match = [i for i, (cd, cr) in enumerate(cells)
                 if min(abs(d - cd), n_pulses - abs(d - cd)) <= tol[0] and abs(r - cr) <= tol[1]]
        if match:
            found.update(match)
        else:
            false += 1
    return len(found), false


def axes(cfg: RadarConfig, n_pulses):
    rng_m = np.arange(cfg.pri_samples) * cfg.range_bin
    vel = (np.arange(n_pulses) - n_pulses // 2) / (n_pulses * cfg.pri) * cfg.wavelength / 2
    return rng_m, vel
