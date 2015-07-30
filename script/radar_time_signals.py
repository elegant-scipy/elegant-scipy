# --- To be ignored by reader ---
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from os.path import join as pjoin

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = '/tmp/radar_time_signals.png'

# Radar parameters
fs = 78125
Teff = 2048.0/fs
Beff = 100E6
S = Beff/Teff

R = 100
fd = 2 * S * R / 3E8

t = np.arange(2048) * 1 / 78125.0
v1 = np.cos(2 * pi * fd * t + 0 * pi/2)
v5 = v1 / 3.0 + 0.2 * np.cos(2 * pi * fd * 1.37 * t + pi / 2) + 0.9 * \
     np.cos(2 * pi * fd * 1.54 * t + pi / 3)
v5 = v5 + 0.02 * np.cos(2 * pi * fd * 1.599 * t + pi / 5) + \
     0.1 * np.cos(2 * pi * fd * 1.8 * t + pi / 6)

stp = np.load(pjoin(os.path.dirname(__file__),
                    '../data/radar_scan_0.npz'))['scan']

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(15, 7))
ax0.plot(1000 * t, v1)
ax0.set_ylabel("$v_1 (t)$, V")
ax0.set_ylim(-1.1, 1.1)
ax0.set_xlim(0, Teff * 1000)
ax0.grid()

ax1.plot(t * 1000, v5)
ax1.set_ylabel("$v_5 (t)$, V")
ax1.set_xlim(0, Teff * 1024)
ax1.set_ylim(-1.5, 1.5)
ax1.grid()

ax2.plot(t * 1000, stp['samples'][5, 14, :] / 8192.0)
ax2.set_ylim(-1.25, 1.25)
ax2.set_xlim(0, Teff * 1000)
ax2.grid()
ax2.set_xlabel("Time, ms")
ax2.set_ylabel("$v_{real}(t)$, V")

plt.savefig(fn, dpi=600)

plt.show()
