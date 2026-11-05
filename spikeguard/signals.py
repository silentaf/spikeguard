"""Transmit waveforms, target echoes and jammer models (all complex baseband, noise power = 1)."""
import numpy as np

from .config import RadarConfig


def _bandlimit(x, fs, half_bw, center=0.0, axis=-1):
    """Ideal FFT-domain band-pass around `center`."""
    n = x.shape[axis]
    f = np.fft.fftfreq(n, 1 / fs)
    mask = np.abs(f - center) <= half_bw
    shape = [1] * x.ndim
    shape[axis] = n
    return np.fft.ifft(np.fft.fft(x, axis=axis) * mask.reshape(shape), axis=axis)


def _unit_noise(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def lfm_pulse(cfg: RadarConfig):
    n = cfg.pulse_samples
    T = n / cfg.fs
    t = np.arange(n) / cfg.fs - T / 2
    return np.exp(1j * np.pi * (cfg.bandwidth / T) * t**2)


def coded_pulse(cfg: RadarConfig, rng):
    """Random quadriphase code, band-limited to the sub-band, unit average power."""
    spc = cfg.pulse_samples // cfg.chips
    p = np.repeat(np.exp(1j * np.pi / 2 * rng.integers(0, 4, cfg.chips)), spc)
    p = _bandlimit(p, cfg.fs, 0.55 * cfg.bandwidth)
    return p / np.sqrt(np.mean(np.abs(p) ** 2))


def make_pulses(cfg: RadarConfig, n_pulses, mode, subband, rng):
    """Return (n_pulses + 1, pulse_samples) waveforms; row 0 is the pulse sent just before the window.

    mode: "fixed" (same LFM every pulse) or "coded" (new random code every pulse).
    """
    k = np.arange(cfg.pulse_samples)
    shift = np.exp(2j * np.pi * cfg.subbands[subband] * k / cfg.fs)
    if mode == "fixed":
        return np.tile(lfm_pulse(cfg) * shift, (n_pulses + 1, 1))
    if mode == "coded":
        return np.stack([coded_pulse(cfg, rng) * shift for _ in range(n_pulses + 1)])
    raise ValueError(mode)


def _add_replica(rx, pulses_used, delay, doppler, amp, cfg):
    """Add amp * pulse[n] delayed by `delay` samples with Doppler phase progression."""
    n_p = rx.shape[0]
    L = cfg.pulse_samples
    if delay < 0 or delay + L > cfg.pri_samples:
        return
    phase = np.exp(2j * np.pi * doppler * np.arange(n_p) * cfg.pri)
    rx[:, delay:delay + L] += amp * phase[:, None] * pulses_used


def target_echoes(cfg, pulses, targets, rng):
    """targets: list of (range_m, velocity_mps, snr_db_per_sample)."""
    n_p = pulses.shape[0] - 1
    rx = np.zeros((n_p, cfg.pri_samples), complex)
    for r, v, snr_db in targets:
        amp = 10 ** (snr_db / 20) * np.exp(2j * np.pi * rng.random())
        _add_replica(rx, pulses[1:], cfg.range_to_delay(r), cfg.vel_to_doppler(v), amp, cfg)
    return rx


JAM_DEFAULTS = dict(
    spot_bw=1.2,         # spot noise width, x radar bandwidth
    spot_offset=0.0,     # Hz, spot centre error relative to the radar sub-band
    sweep_periods=3.0,   # PRIs per sweep across the band
    sweep_bw=4e6,        # Hz, instantaneous width of the swept noise
)


def random_jam_params(cfg, rng):
    return dict(spot_bw=rng.uniform(0.8, 2.0), spot_offset=rng.uniform(-3e6, 3e6),
                sweep_periods=rng.uniform(1, 8), sweep_bw=rng.uniform(1e6, 8e6))


def jammer(cfg, kind, jnr_db, pulses, subband, rng, drfm_targets=None, params=None):
    """Jamming signal at the radar receiver, (n_pulses, pri_samples).

    jnr_db is jammer power per sample relative to receiver noise. The DRFM replays the
    previously captured pulse (it cannot know a new random code before it is sent).
    """
    prm = {**JAM_DEFAULTS, **(params or {})}
    n_p = pulses.shape[0] - 1
    shape = (n_p, cfg.pri_samples)
    p = 10 ** (jnr_db / 10)
    if kind == "clean":
        return np.zeros(shape, complex)
    if kind == "barrage":
        return np.sqrt(p) * _unit_noise(rng, shape)
    if kind == "spot":
        j = _bandlimit(_unit_noise(rng, shape), cfg.fs, prm["spot_bw"] * cfg.bandwidth / 2,
                       cfg.subbands[subband] + prm["spot_offset"])
        return np.sqrt(p) * j / np.sqrt(np.mean(np.abs(j) ** 2))
    if kind == "sweep":
        n = n_p * cfg.pri_samples
        nb = _bandlimit(_unit_noise(rng, n), cfg.fs, prm["sweep_bw"] / 2)
        period = int(prm["sweep_periods"] * cfg.pri_samples)
        t0 = rng.integers(period)
        frac = ((np.arange(n) + t0) % period) / period
        f_inst = -0.5 * cfg.fs * 0.95 + 0.95 * cfg.fs * frac
        j = nb * np.exp(2j * np.pi * np.cumsum(f_inst) / cfg.fs)
        j = np.sqrt(p) * j / np.sqrt(np.mean(np.abs(j) ** 2))
        return j.reshape(shape)
    if kind == "drfm":
        rx = np.zeros(shape, complex)
        for r, v in drfm_targets:
            amp = np.sqrt(p) * np.exp(2j * np.pi * rng.random())
            _add_replica(rx, pulses[:-1], cfg.range_to_delay(r), cfg.vel_to_doppler(v), amp, cfg)
        return rx
    raise ValueError(kind)


def random_drfm_targets(cfg, rng, k=3):
    return [(rng.uniform(450, 2700), rng.uniform(-0.8, 0.8) * cfg.vel_max) for _ in range(k)]


def receive(cfg, n_pulses, mode, subband, targets, jam_kind, jnr_db, rng,
            drfm_targets=None, jam_subband=None, jam_params=None):
    """Simulate one receive window. Returns (rx, reference_pulses).

    jam_subband: band a spot jammer sits on (defaults to the radar's band, i.e. a set-on jammer
    that reacted to the previous CPI).
    """
    if jam_subband is None:
        jam_subband = subband
    pulses = make_pulses(cfg, n_pulses, mode, subband, rng)
    if jam_kind == "drfm" and drfm_targets is None:
        drfm_targets = random_drfm_targets(cfg, rng)
    rx = target_echoes(cfg, pulses, targets, rng)
    rx += jammer(cfg, jam_kind, jnr_db, pulses, jam_subband, rng, drfm_targets, jam_params)
    rx += _unit_noise(rng, rx.shape)
    return rx, pulses[1:]
