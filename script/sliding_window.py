import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.io import wavfile

import sys

# Read audio file released under CC BY 4.0 at
# http://www.orangefreesounds.com/nightingale-sound/

# Get sampling rate (number of measurements per second)
# as well as audio data as an (N, 2) array -- two columns
# because this is a stereo recording.

rate, audio = wavfile.read('data/nightingale.wav')
left, right = audio.T

plt.style.use('style/thinner.mplstyle')

f, ax = plt.subplots(figsize=(4.8, 2.4))
ax.plot(left[:1524], zorder=-100)

window_0 = Rectangle((0, -80), 1024, 160, fill=False, lw=1)
window_1 = Rectangle((100, -85), 1024, 155, fill=False, lw=1, ls='--')
ax.add_patch(window_0)
ax.add_patch(window_1)

plt.annotate(
    '', xy=(0, -60), xycoords='data',
    xytext=(1024, -60), textcoords='data',
    arrowprops={'arrowstyle': '<->',
                'linewidth': 1})

plt.annotate(
    '1024', xy=(1024 / 2, -60), xycoords='data',
    xytext=(-10, -8), textcoords='offset points',
    size='small')

plt.annotate(
    '', xy=(1024, -70), xycoords='data',
    xytext=(1124, -70), textcoords='data',
    arrowprops={'arrowstyle': '<-',
                'linewidth': 1})

plt.annotate(
    '100', xy=(1074, -70), xycoords='data',
    xytext=(-8, 5), textcoords='offset points',
    size='small')


ax = plt.gca()
for spine in ('right', 'top', 'bottom', 'left'):
    ax.spines[spine].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(-90, 90)
ax.set_xlim(-1, 1530)

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = '/tmp/sliding_window.png'

plt.savefig(fn, dpi=300)

#plt.show()
