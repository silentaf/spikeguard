"""Per-pulse feature vectors for the SNN: one time step = one PRI, so the network streams pulse by pulse.

Each PRI gives 96 features (log power / 10, so receiver noise ~ 0):
  32 spectral bins   - where in frequency the energy is (spot / sweep / barrage)
  32 temporal bins   - pulsed vs continuous energy within the PRI (DRFM vs noise)
  32 range bins      - matched-filter peaks (coherent false targets)
"""
import numpy as np

from .processing import pulse_compress

N_BINS = 32
N_FEATURES = 3 * N_BINS


def _db(x):
    return np.log10(x + 1e-12)


def pulse_features(rx, refs):
    n_p, n = rx.shape
    spec = np.abs(np.fft.fftshift(np.fft.fft(rx, 2048, axis=1), axes=1)) ** 2 / n
    spec = spec.reshape(n_p, N_BINS, -1).mean(axis=2)

    seg = n // N_BINS
    temporal = (np.abs(rx[:, :seg * N_BINS]) ** 2).reshape(n_p, N_BINS, seg).mean(axis=2)

    mf = np.abs(pulse_compress(rx, refs)) ** 2
    rangef = mf[:, :seg * N_BINS].reshape(n_p, N_BINS, seg).max(axis=2) / np.log(seg)

    return np.concatenate([_db(spec), _db(temporal), _db(rangef)], axis=1).astype(np.float32)
