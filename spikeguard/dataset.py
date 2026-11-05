"""Self-labelling jammer dataset: the jammer knows its own mode, so every capture is labelled for free.

On hardware the same generator drives the jammer channel and the radar channel records; here both
are simulated. Each sample is (T pulses x N_FEATURES), label = index into JAMMERS.
"""
from multiprocessing import Pool

import numpy as np

from .config import RadarConfig, JAMMERS
from . import signals
from .features import pulse_features

T_STEPS = 16


def random_scene(cfg, rng):
    """Randomise everything the classifier should be invariant to."""
    targets = [(rng.uniform(450, 2600), rng.uniform(-0.8, 0.8) * cfg.vel_max, rng.uniform(-18, -8))
               for _ in range(rng.integers(0, 3))]
    return dict(
        mode=str(rng.choice(["fixed", "coded"])),
        subband=int(rng.integers(len(cfg.subbands))),
        targets=targets,
        jnr=float(rng.uniform(-10, 30)),
        gain_db=float(rng.uniform(-2, 2)),            # receiver gain / noise-floor calibration error
        jam_params=signals.random_jam_params(cfg, rng),
        drfm_targets=signals.random_drfm_targets(cfg, rng, int(rng.integers(1, 6))),
    )


def make_sample(cfg, label, rng, scene=None):
    scene = scene or random_scene(cfg, rng)
    rx, refs = signals.receive(cfg, T_STEPS, scene["mode"], scene["subband"], scene["targets"],
                               JAMMERS[label], scene["jnr"], rng, drfm_targets=scene["drfm_targets"],
                               jam_params=scene["jam_params"])
    rx *= 10 ** (scene["gain_db"] / 20)
    return pulse_features(rx, refs), scene["jnr"]


def _worker(args):
    label, seed = args
    cfg = RadarConfig()
    x, jnr = make_sample(cfg, label, np.random.default_rng(seed))
    return x, label, jnr


def generate(n_per_class, seed=0):
    jobs = [(c, seed * 10_000_000 + i * len(JAMMERS) + c)
            for i in range(n_per_class) for c in range(len(JAMMERS))]
    with Pool() as pool:
        rows = pool.map(_worker, jobs, chunksize=64)
    X = np.stack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    jnr = np.array([r[2] for r in rows], np.float32)
    return X, y, jnr
