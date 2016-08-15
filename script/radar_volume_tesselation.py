import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy import spatial

import os
from os.path import join as pjoin

from colormaps import _inferno_data


v = np.load('../data/radar_scan_1.npz')['scan']['samples']
V = np.fft.fft(v, axis=2)[::-1, :, :v.shape[2] // 2]

V = np.abs(V)
V = 20 * np.log10(V / V.max())
V -= V.min()
V /= V.max()

r = np.argmax(V, axis=2)

el, az = np.meshgrid(*[np.arange(s) for s in r.shape], indexing='ij')

coords = np.column_stack((el.flat, az.flat, r.flat))
axis_labels = ['Elevation', 'Azimuth', 'Range']

d = spatial.Delaunay(coords[:, :2])
simplexes = coords[d.vertices]

coords = np.roll(coords, shift=-1, axis=1)
axis_labels = np.roll(axis_labels, shift=-1)

f, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
ax.plot_trisurf(*coords.T, triangles=d.vertices,
                cmap='magma_r')

ax.set_xlabel(axis_labels[0])
ax.set_ylabel(axis_labels[1])
ax.set_zlabel(axis_labels[2])
ax.view_init(azim=-45)

plt.show()
