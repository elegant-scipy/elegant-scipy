# --- To be ignored by reader ---
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from os.path import join as pjoin

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = '/tmp/radar_time_signals.png'


pi = np.pi

# Radar parameters
fs = 78125          # Sampling frequency in Hz, i.e. we sample 78125
                    # times per second

ts = 1 / fs         # Sampling time, i.e. one sample is taken each
                    # ts seconds

Teff = 2048.0 * ts  # Total sampling time for 2048 samples
                    # (AKA effective sweep duration) in seconds.

Beff = 100e6        # Effective bandwidth in Hz
S = Beff / Teff     # Frequency sweep rate in Hz/s

R = 100             # Simulated target range

fd = 2 * S * R / 3E8  # Frequency difference

t = np.arange(2048) * ts  # Sample times

# Generate five targets
v0 = np.cos(2 * pi * fd * t)
v1 = np.cos(2 * pi * fd * 1.37 * t + pi / 2)
v2 = np.cos(2 * pi * fd * 1.54 * t + pi / 3)
v3 = np.cos(2 * pi * fd * 1.599 * t + pi / 5)
v4 = np.cos(2 * pi * fd * 1.8 * t + pi / 6)

# Blend them together
v5 = (v1 / 3.0) + (0.2 * v1) + (0.9 * v2) + (0.02 * v3) + (0.1 * v4)

plt.plot(t, v5)
plt.show()


scan = np.load(pjoin(os.path.dirname(__file__),
               '../data/radar_scan_0.npz'))['scan']

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(15, 7))
ax0.plot(1000 * t, v0)
ax0.set_ylabel("$v_{single} (t)$, V")
ax0.set_ylim(-1.1, 1.1)
ax0.set_xlim(0, Teff * 1000)
ax0.grid()

ax1.plot(t * 1000, v5)
ax1.set_ylabel("$v_{sim} (t)$, V")
ax1.set_xlim(0, Teff * 1024)
ax1.set_ylim(-1.5, 1.5)
ax1.grid()

ax2.plot(t * 1000, scan['samples'][5, 14, :] / 8192.0)
ax2.set_ylim(-1.25, 1.25)
ax2.set_xlim(0, Teff * 1000)
ax2.grid()
ax2.set_xlabel("Time, ms")
ax2.set_ylabel("$v_{real}(t)$, V")

plt.savefig(fn, dpi=300)

plt.show()
