```python
%matplotlib inline
```

# The Fast Fourier Transform (FFT)

*This chapter was written in collaboration with SW's father, PW van der Walt.*

When taking measurements (or recording a "signal"), we usually do so
along regular intervals in time or space.  For example, an image
represents intensity values that vary across space, whereas a sound
recording tells us how sound intensity changes over time.

Imagine, now, that you are an intra-dimensional being that could see
all of time at once (yes, that would be awesome!).  This would allow
you to gain one new kind of insight among many: which *frequencies*
("notes" in terms of music, "texture" in terms of space) are included
in a given signal.  The tool that allows mere mortals to gain access
to this information is called the Fourier Transform.

Sometimes, we want to position ourselves somewhere in the middle of
the time and Fourier domains: we want to know not only what
frequencies are included in a signal, but also *when* they occur.
Sadly, a variation of the uncertainty principle constrains us in the
accuracy within which we can pin-point both simultaneously (and that
also makes sense logically: a low frequency component changes slowly,
it would be hard to say exactly when it occurs, whereas a high
frequency component is much easier to place).  The wavelet transform
(the SciPy implementation of which is well underway) makes such a
compromise and gives you a "best of both worlds" transform.

Returning to Fourier Transform, tracing its exact origins proves to be
tricky.  The use of harmonic series dates back to Babylonian times,
but it was the hot topics of calculating asteroid orbits and solving
the heat equation that led to several breakthroughs in the early
1800s.  Whom exactly among Clairaut, LaGrange, Euler, Gauss and
D'Alembert we should thank is not exactly clear, but we know that
Gauss first came up with the Fast Fourier Transform (an algorithm for
computing the Discrete Fourier Transform, popularized by Cooley and
Tukey in 1965).  Furthermore it is believed that Fourier first claimed
that abitrary functions could be expressed by a trigonometric
(Fourier) series.

The Fourier Transform functionality in SciPy lives in the fairly
spartan ``scipy.fftpack`` module.  It provides the following
FFT-related functionality:

 - ``fft``, ``fft2``, ``fftn``: Compute the Fast (discrete) Fourier Transform
 - ``ifft``, ``ifft2``, ``ifftn``: Compute the inverse of the FFT
 - ``dct``, ``idct``, ``dst``, ``idst``: Compute the cosine and sine transforms
 - ``fftshift``, ``ifftshift``: Shift the zero-frequency component to the center of the
   spectrum and back, respectively (more about that soon)
 - ``fftfreq``: Return the Discrete Fourier Transform sample frequencies

This is complemented by the following functions in NumPy:

 - ``np.hanning``, ``np.hamming``, ``np.bartlett``, ``np.blackman``,
   ``np.kaiser``: Tapered windowing functions.

SciPy wraps FFTPACK as its underlying implementation--it is not the
fastest out there, but unlike packages such as FFTW, it has a
permissive free software license.

Consider that a naive calculation of the FFT takes
$\mathcal{O}\left(N^2\right)$ operations, whereas the Fast Fourier
Transform is $\mathcal{O}(N \log N)`` in the ideal case--a great
improvement!  However, the classical Cooley-Tukey algorithm
implemented in FFTPACK recursively breaks up the transform into
smaller (prime-sized) pieces and only shows this improvement for
highly smooth input lengths (an input length is considered smooth when
its largest prime factor is small).  For large prime sized pieces, the
Bluestein or Rader algorithms can be used in conjunction with the
Cooley-Tukey algorithm, but this optimization is not implemented in
FFTPACK.

Let us illustrate:

```python
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy import fftpack
from sympy import factorint

K = 1000
lengths = range(250, 260)

# Calculate the smoothness for all input lengths
smoothness = [max(factorint(i).keys()) for i in lengths]


exec_times = []
for i in lengths:
    z = np.random.random(i)

    # For each input length i, execute the FFT K times
    # and store the execution time

    times = []
    for k in range(K):
        tic = time.monotonic()
        fftpack.fft(z)
        toc = time.monotonic()
        times.append(toc - tic)

    # For each input length, remember the *minimum* execution time
    exec_times.append(min(times))


f, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
ax0.stem(lengths, exec_times)
ax0.set_xlabel('Length of input')
ax0.set_ylabel('Execution time (seconds)')

ax1.stem(lengths, smoothness)
ax1.set_ylabel('Smoothness of input length')

plt.show()
```

If you are able to use software with different licenses, you may
consider using [PyFFTW](https://github.com/hgomersall/pyFFTW),
[Reikna](https://github.com/fjarri/reikna) or
[NumbaPro](http://docs.continuum.io/numbapro/cudalib.html).

Next, we will present a couple of common concepts worth knowing before
operating heavy Fourier Transform machinery, whereafter we tackle a
real-world problem: analyzing target detection in radar data.

## Discrete Fourier Transform concepts

When executing the FFT, the returned spectrum (collection of
frequencies, or Fourier components) is circular, and packed from
low-to-high-to-low.  E.g., when we do the real Fourier transform of
all ones, an input that has no variation and therefore only has the
slowest, constant Fourier component (also known as the "DC"--for
direct current--component), that component appears as the first entry:

```python
from scipy import fftpack
fftpack.fft(np.ones(10))
```

Note that the FFT returns a complex spectrum which, in the case of
real inputs, is symmetrical.

When we try the FFT on a rapidly changing signal, we see a high
frequency component appear:

```python
z = np.ones(10)
z[::2] = -1

print("Applying FFT to {}".format(z))
fftpack.fft(z)
```

The `fftfreq` function tells us which frequencies we are looking at:

```
fftpack.fftfreq(10)
```

The result tells us that our maximum component occured at a frequency
of 0.5 cycles per sample.  This agrees with the input, where a
minus-one-plus-one cycle repeated every second sample.

Sometimes, it is convenient to view the spectrum organized slightly
differently, from high-to-low-to-high.  This is achieved with the
`fftshift` function.  Let's examine the frequency components in a
noisy image:

(NOTE: get permission to reproduce the image below
https://miac.unibas.ch/SIP/06-Restoration.html#(16), or even better
find an alternative or reproduce)

```python
from skimage import io
image = io.imread('images/pompei.jpg')

print(image.shape, image.dtype)

F = fftpack.fftn(image)
F_magnitude = np.abs(F)
F_magnitude = fftpack.fftshift(F_magnitude)

f, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 15))
ax0.imshow(image, cmap='gray')
ax1.imshow(-np.log(1 + F_magnitude), cmap='gray')
plt.show()
```

- Windowing introduction (no detail given)
- Frequencies used (fftfreq)

...

- fft/ifft (1D) [timing on real vs complex sequence -> rfft
- dct (1D)
  - compressed sensing
    http://www.mathworks.com/company/newsletters/articles/clevescorner-compressed-sensing.html

- fft2/ifft2 (2D)
  - explain fftshift to get spectrum in expected form
  - example: image notch filter
    https://miac.unibas.ch/SIP/06-Restoration.html
  - spectral method to solve, e.g., Poisson equation
    See arXiv:1111.4971, "On FFT-based convolutions and correlations, with
    application to solving Poisson’s equation in an open
    rectangular pipe" by R. Ryne

- Radar example
  - In active discussion with source

- fftn/ifftn (3D)
 - Phase correlation from skimage

- convolution types: numpy, ndimage, signal.convolve, signal.fftconvolve
