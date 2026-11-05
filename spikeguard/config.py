"""Radar parameters, chosen to fit one USRP N310 (4 coherent channels, ~100 MHz bandwidth)."""
from dataclasses import dataclass

C = 299_792_458.0


@dataclass(frozen=True)
class RadarConfig:
    fs: float = 100e6            # complex baseband sample rate
    fc: float = 2.45e9           # RF centre frequency (cabled, so any legal LO works)
    pri_samples: int = 2000      # 20 us PRI -> 50 kHz PRF
    pulse_samples: int = 200     # 2 us pulse
    bandwidth: float = 20e6      # instantaneous waveform bandwidth per sub-band
    subbands: tuple = (-37.5e6, -12.5e6, 12.5e6, 37.5e6)  # digital agility inside the 100 MHz IF
    n_pulses: int = 128          # pulses per CPI
    chips: int = 40              # phase-code length in coded (diversity) mode
    pfa: float = 1e-7            # CFAR design Pfa; range-correlated noise makes the measured rate ~1e-6

    @property
    def pri(self):
        return self.pri_samples / self.fs

    @property
    def wavelength(self):
        return C / self.fc

    @property
    def range_bin(self):
        return C / (2 * self.fs)

    @property
    def range_res(self):
        return C / (2 * self.bandwidth)

    @property
    def vel_res(self):
        return self.wavelength / (2 * self.n_pulses * self.pri)

    @property
    def vel_max(self):
        return self.wavelength / (4 * self.pri)

    def range_to_delay(self, r):
        return int(round(r / self.range_bin))

    def vel_to_doppler(self, v):
        return 2 * v / self.wavelength


JAMMERS = ("clean", "barrage", "spot", "sweep", "drfm")
