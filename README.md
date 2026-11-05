# SpikeGuard — neuromorphic jammer recognition and adaptive ECCM on a USRP radar

IEEE Radar Challenge 2027 entry (themes 5.1 + 5.5 + 5.6 + 5.7).

A pulse-Doppler radar and a DRFM-style jammer share one 4-channel USRP N310, connected **only by
cables and attenuators**, with nothing transmitted over the air. A small spiking neural network (SNN)
watches the received signal pulse by pulse, recognises the jamming type, and the radar switches to
the matching electronic counter-countermeasure (ECCM) for the next coherent processing interval (CPI).

```
 Ch A radar TX ─► splitter ─► attenuator ─► combiner ─► Ch B radar RX ─► pulse compression ─► range-Doppler ─► CFAR
                     │                         ▲             │
                     ▼                         │             └─► features ─► SNN ─► jammer class ─► ECCM policy
 Ch C DRFM capture ◄─┘          Ch D jammer TX ┘                                                     │
                                                         next CPI: hop band / excise / random codes ◄┘
```

## Current status: simulation (pre-hardware)

Everything below runs in Python and mirrors the hardware plan, so the same processing code will run
on the recorded USRP data.

| Item | Value |
|---|---|
| Waveform | 2 µs pulse, 20 MHz, PRI 20 µs, 128 pulses/CPI, 4 digital sub-bands inside 100 MHz |
| Resolution | 7.5 m range, 23.9 m/s velocity, ±1530 m/s unambiguous |
| Jammers | barrage noise, spot noise, swept noise, DRFM false targets (replays previous pulse) |
| ECCM | spot → frequency hop to least-jammed sub-band · sweep → time-frequency excision · DRFM → per-pulse random phase codes · barrage → none (honest limit: only processing gain helps) |
| SNN | 96-128-64-5 LIF, 20.8k synapses, one time step per pulse |

### Results

| Metric | Result |
|---|---|
| SNN jammer recognition, all JNR −10…30 dB | **92.8 %** (MLP baseline 94.8 %) |
| SNN recognition, JNR ≥ 5 dB | **100 %** (errors only below the noise floor) |
| int8 weights (FPGA-ready) | 92.6 % (−0.2 %) |
| Hidden-layer spike rate | **1.9 %** of neurons per step (sparse means cheap in FPGA logic) |
| Decision latency | ~89 % accuracy after 6 pulses (120 µs), 93 % after 16 (320 µs) |
| Spot jammer, JNR 15–30 dB | Pd 0.03 → **1.00** with SNN-driven hopping |
| Sweep jammer, JNR 20–30 dB | Pd ≈ 0.02 → **≥0.95** with excision |
| DRFM | false targets 3.4 → **≈0.1 per CPI** with random codes; cost: Pd drops to 0.5–0.65 at JNR ≥ 15 dB from decorrelated replay energy |
| SNN-driven vs oracle ECCM | indistinguishable within Monte Carlo error |

Figures: `figures/rd_gallery.png`, `figures/pd_vs_jnr.png`, `figures/snn_confusion.png`,
`figures/snn_accuracy.png`, `figures/closed_loop.png`.

Known limitations (to be checked on hardware): simulated jammers share the radar clock; no IQ
imbalance, phase noise or TX→RX leakage yet; the DRFM replays with integer-sample delay.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/eccm_study.py     # range-Doppler gallery + Pd vs JNR (~1 min)
python scripts/train_snn.py      # dataset (12k samples) + SNN/MLP training (~2 min on GPU)
python scripts/closed_loop.py    # SNN in the loop vs no ECCM vs oracle (~1 min)
```

## Layout

```
spikeguard/   config, signals (waveforms, targets, jammers), processing (MF, Doppler, CFAR, excision),
              eccm (policy), features (per-pulse SNN inputs), dataset (self-labelled), snn (model)
scripts/      experiments that produce figures/ and results/
```

## Roadmap

1. Jan 2027: UHD/GNU Radio bring-up, cable-delay ranging baseline (challenge milestone).
2. Feb: radar + timed-TX jammer on N310 channels; record a labelled hardware dataset.
3. Mar: retrain the SNN on hardware captures; closed loop on host.
4. Stretch: SNN as an RFNoC block (VHDL LIF neurons, int8 weights in BRAM); low-latency DRFM in FPGA.

Licence: MIT.
