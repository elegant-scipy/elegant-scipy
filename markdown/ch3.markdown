```python
%matplotlib inline

import seaborn as sns
sns.set_style('white')
sns.despine()

import matplotlib.pyplot as plt
if hasattr(plt.cm, 'inferno'):
    plt.rcParams['image.cmap'] = 'inferno'
```

<!--

## Notes

- Windowing introduction (no detail given)

- fft/ifft (1D) [timing on real vs complex sequence -> rfft
- dct (1D)
  - compressed sensing
    http://www.mathworks.com/company/newsletters/articles/clevescorner-compressed-sensing.html

- fft2/ifft2 (2D)
  - example: image notch filter
    https://miac.unibas.ch/SIP/06-Restoration.html
  - spectral method to solve, e.g., Poisson equation
    See arXiv:1111.4971, "On FFT-based convolutions and correlations, with
    application to solving Poisson’s equation in an open
    rectangular pipe" by R. Ryne

- fftn/ifftn (3D)
 - Phase correlation from skimage

- convolution types: numpy, ndimage, signal.convolve, signal.fftconvolve

-->

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

While ideally we don't want to reimplement existing algorithms,
sometimes it becomes necessary in order to obtain the best execution
speeds possible, and tools like [Cython](http://cython.org)—which
compiles Python to C—and [Numba](http://numba.pydata.org)—which
does just-in-time compilation of Python code—make life a lot easier
(and faster!).

If you are able to use GPL-licenced software, you may
consider using [PyFFTW](https://github.com/hgomersall/pyFFTW) for
faster FFTs.

Next, we present a couple of common concepts worth knowing before
operating heavy Fourier Transform machinery, whereafter we tackle a
real-world problem: analyzing target detection in radar data.

## Discrete Fourier Transform concepts

When executing the FFT, the returned spectrum (collection of
frequencies, or Fourier components) is circular, and packed from
low-to-high-to-low.  E.g., when we do the real Fourier transform of a
signal of all ones, an input that has no variation and therefore only
has the slowest, constant Fourier component (also known as the
"DC"--for direct current--component), that component appears as the
first entry:

```python
from scipy import fftpack
fftpack.fft(np.ones(10))
```

Note that the FFT returns a complex spectrum which, in the case of
real inputs, is conjugate symmetrical.

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
differently, from high-negative to low to-high-positive (for now, we
won't dive too deeply into the concept of negative frequency, other
than saying a real-world sine wave is produced by a combination of
positive and negative frequencies).  We re-shuffle the spectrum using
the `fftshift` function.  Let's examine the frequency components in a
noisy image:

```python
from skimage import io
image = io.imread('images/moonlanding.png')
M, N = image.shape

print((M, N), image.dtype)

F = fftpack.fftn(image)
F_magnitude = np.abs(F)
F_magnitude = fftpack.fftshift(F_magnitude)

f, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 15))
ax0.imshow(image, cmap='gray')
ax1.imshow(-np.log(1 + F_magnitude),
           cmap='gray', interpolation='nearest',
           extent=(-N // 2, N // 2, -M // 2, M // 2))
plt.show()
```

Note the high values around the origin (middle) of the spectrum—these
coefficients describe the low frequencies or smooth parts of the
image; a vague canvas of the photo.  Higher frequency components,
spread throughout the spectrum, fill in the edges and detail.  Peaks
around higher frequencies correspond to the periodic noise.

## Real-world Application: Analyzing Radar Data

Linearly modulated FMCW (Frequency-Modulated Continuous-Wave) radars
make extensive use of the FFT algorithm for signal processing and
provide examples of various application of the FFT. In this chapter we
will use actual data from FMCW radars to demonstrate one such
application: target detection.

To start off, we take a bird's eye tour of FMCW radars so we may
better understand the properties of the signals involved.

### Block diagram of a simple FMCW radar

A block diagram of a simple FMCW radar that uses separate transmit and
receive antennas is shown in Fig. [fig: block-diagram]. The radar
consists of a waveform generator that generates a linearly frequency
modulated sinusoidal signal at the required transmit frequency. A
Direct Digital Synthesizer (DDS) controlled by the computer is often
used in modern systems. The output frequency of the DDS signal is
converted to the desired radio frequency. The generated signal is
amplified to the required power level by the transmit amplifier and
routed to the transmit antenna via a coupler circuit where an
amplitude scaled copy of the transmit signal is tapped off. The
transmit antenna radiates the transmit signal as an electromagnetic
wave in a narrow beam towards the target to be detected. When the wave
encounters an object that reflects electromagnetic waves, a fraction
of of the energy irradiating the target is reflected back to the
receiver as a second electromagnetic wave that propagates in the
direction of the radar system. When this wave encounters the receive
antenna, the antenna collects the energy in the wave energy impinging
on it and converts this to a fluctuating voltage that is fed to the
mixer. The mixer multiplies the received signal with a replica of the
transmit signal and produces a sinusoidal signal with a frequency
equal to the difference in frequency between the transmitted and
received signals. The low-pass filter ensures that the received signal
is band limited (i.e., does not contain frequencies that we don't care
about) and the receive amplifier strengthens the signal to a suitable
amplitude for the analog to digital converter (ADC) that feeds data to
the computer. The data consists of $N$ samples sampled at a frequency
$f_{s}$.

![[fig: block-diagram]The block diagram of a simple FMCW radar system.](../figures/FMCW Block.png)

### Signal properties in the time domain

The transmit signal is a sinusoidal signal with an instantaneous
frequency that increases linearly with time, as shown in
Fig. [fig:FMCW waveform](a).

Starting at $f_{min}$, the frequency increases at a rate $S$ Hz/s to
$f_{max}.$ The frequency is then decreased rapidly back to $f_{min}$
after which a next frequency sweep occurs.

![[fig:FMCW waveform]The frequency relationships in an FMCW radar with
 linear frequency modulation.](../figures/FMCW waveform.png)

This signal is radiated by a directional transmit antenna. When the
wave with propagation velocity $v\approx300\times10^{6}$ m/s (the
propagation speed of electro-magnetic waves in air is ever-so-slightly
slower than the speed of light in a vacuum) hits a target at a range
$R$, the echo will reach the radar after a time

$$t_{d}=\frac{2R}{v}.\label{eq:transit time}$$

Here it is collected by the receive antenna and converted to a
sinusoidally fluctuating voltage. The received signal is a replica of
the transmitted signal, delayed by the transit time $t_{d}$ and is
shown dashed in Fig. [fig:FMCW waveform](b).

A radar is designed to detect targets up to a maximum range
$R_{max}$. Echoes from maximum range reach the radar after a transit
time $t_{dm}$ as shown in Fig. [fig:FMCW waveform](c).

We note that there is a constant difference in frequency between the
transmitted and received signals and this will be true for all targets
after time $t_{s}$ until $t_{e}$. We conclude from
Fig. [fig:FMCW waveform] that the frequency difference is given by

$$f_{d}=S\times t_{d}=\frac{2SR}{v}\label{eq:difference frequency}$$

$T_{eff}=t_{e}-t_{s}=\frac{N}{f_{s}}$ is the effective sweep duration
of the radar. The frequency excursion of the sweep during $T_{eff}$ is
the effective bandwidth of the radar, given by

$$B_{eff}=f_{max}-f_{1}=ST_{eff}.\label{eq:Effective bandwidth}$$

We will see that the range resolution of the radar is determined by
the effective bandwidth.

Returning to Fig. [fig: block-diagram], the signal produced by the
receiver at the input to the Analog to Digital Converter (ADC) when
there is a single target is a sinusoid with constant amplitude,
proportional to the amplitude of the echo, and constant frequency,
proportional to the range to the target.

Such a signal is shown as $v_{1}(t)$ in Fig. [fig:radar time signals]
for a radar with parameters $B_{eff}=100$ MHz, sampling frequency
28125 Hz and N=2048 samples. The effective sweep time is
$T_{eff}=\frac{2048}{28125}=26.214$ ms. We can interpret this signal
by counting the number of cycles in the graph — about
$66\frac{1}{2}$. The difference frequency is approximately
$\frac{66.5}{26.214E-3}=6.35$ kHz. With
$S=\frac{B_{eff}}{T_{eff}}=\frac{100E6}{26.214E-3}=3.815\times10^{9}$
Hz/s, we can calculate the range to the target as
$R=\frac{vf_{d}}{2S}=\frac{3\times10^{8}\times6.35\times10^{3}}{2\times3.815\times10^{9}}=249.7$
m.

A real radar will rarely receive only a single echo. The simulated
signal $v_{5}(t)$ shows what a radar signal will look like with 5
targets at different ranges and $v_\mathrm{actual}(t)$ shows the output signal
obtained with an actual radar. We cannot interpret these signals in
the time domain. They just make no sense at all!

![[fig:radar time signals]Receiver output signals (a) single target
 (b) 5 targets (c) actual radar data. ](../figures/generated/radar_time_signals.png)

Apart from the real world signal shown above, the others were
generated as follows:

```python
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

```

The real world radar data is read from a NumPy-format ``.npz`` file (a
light-weight, cross platform and cross-version compatible storage
format).  These files can be saved with the ``np.savez`` or
``np.savez_compressed`` functions.  Note that SciPy's ``io`` submodule
can also easily read other formats, such as MATLAB(R) and NetCDF
files.


```python
data = np.load('data/radar_scan_0.npz')

# Load variable 'scan' from 'radar_scan_0.npz'
scan = data['scan']

# Grab one (azimuth, elevation) measurement
# It has shape (2048,)
v_actual = scan['samples'][5, 14, :]

# Scale v_actual to have a reasonable maximum
print('v_actual has a maximum of {} before normalization'.format(v_actual.max()))

v_actual = v_actual / 3000

print('v_actual has a maximum of {} after normalization'.format(v_actual.max()))

```

Since ``.npz``-files can store multiple variables, we have to select
the one we want: ``data['scan']``.  That returns a
*structured NumPy array* with the following fields:

- **time** : unsigned 64-bit (8 byte) integer (`np.uint64`)
- **size** : unsigned 32-bit (4 byte) integer (`np.uint32`)
- **position** :
  - **az** : 32-bit float (`np.float32`)
  - **el** : 32-bit float (`np.float32`)
  - **region_type** : unsigned 8-bit (1 byte) integer (`np.uint8`)
  - **region_ID** : unsigned 16-bit (2 byte) integer (`np.uint16`)
  - **gain** : unsigned 8-bit (1 byte) integer (`np.uin8`)
  - **samples** : 2048 unsigned 16-bit (2 byte) integers (`np.uint16`)

While it is true that NumPy arrays are *homogeneous* (i.e., all the
elements inside are the same), it does not mean that those elements
cannot be compound elements, as is the case here.

An individual field is accessed using dictionary syntax:

```python
azimuths = scan['position']['az']  # Get all azimuth measurements
```

To construct an array such as the above from scratch, one would first
set up the appropriate dtype:

```python
dt = np.dtype([('time', np.uint64),
               ('size', np.uint32),
               ('position', [('az', np.float32),
               ('el', np.float32),
               ('region_type', np.uint8),
               ('region_ID', np.uint16)]),
               ('gain', np.uint8),
               ('samples', (np.int16, 2048))])
```

The dtype can then be used to create an array, which we can later fill
with values:

```python
data = np.zeros(500, dtype=dt)  # Construct array with 500 measurements
```

To summarize what we've seen so far: the shown measurements ($v_5$ and
$v_\mathrm{actual}$) are the sum of sinusoidal signals generated by each of the
distinct reflectors.  We need to determine each of the constituent
components of these composite radar signals. The FFT is the tool that
will do this for us. After a short introduction to the theory of the
FFT, we show how we interpret the radar data.

### Discrete Fourier transforms

The Discrete Fourier Transform (DFT) converts a sequence of $N$
equally spaced real or complex samples $x_{0,}x_{1,\ldots x_{N-1}}$ of
a function $x(t)$ of time (or another variable, depending on the
application) into a sequence of $N$ complex numbers $X_{k}$ by the
summation

$$X_{k}=\sum_{n=0}^{N-1}x_{n}e^{-j2\pi kn/N},\;k=0,1,\ldots
N-1.\label{eq:Forward DFT}$$

With the numbers $X_{k}$ known, the inverse DFT *exactly* recovers the
sample values $x_{n}$ through the summation

$$x_{n}=\frac{1}{N}\sum_{k=0}^{N-1}X_{k}e^{j2\pi
kn/N}.\label{eq:Inverse DFT}$$

Keeping in mind that $e^{j\theta}=\cos\theta+j\sin\theta,$ the last
equation shows that the DFT has decomposed the sequence $x_{n}$ into a
complex discrete Fourier series with coefficients $X_{k}$. Comparing
the DFT with a continuous complex Fourier series

$$x(t)=\sum_{n=-\infty}^{\infty}c_{n}e^{jn\omega_{0}t},\label{eq:Complex
Fourier series}$$

the DFT is a *finite *series with $N$ terms defined at the equally
spaced discrete instances of the *angle*
$(\omega_{0}t_{n})=2\pi\frac{k}{N}$ in the interval
$[0,2\pi)$, i.e. *including* 0 and *excluding* $2\pi$.
This automatically normalizes the DFT so that time does
not appear explicitly in the forward or inverse transform.

If the original function $x(t)$ is *band limited* to less than half of the
sampling frequency, interpolation between sample values produced by the inverse
DFT will usually (see the discussion below on windowing)
give a faithful reconstruction of $x(t)$. If $x(t)$ is *not* band limited,
the inverse DFT can, in general, not be used to reconstruct $x(t)$ by
interpolation.

The function $e^{j2\pi k/N}=\left(e^{j2\pi/N}\right)^{k}=w^{k}$ takes on
discrete values between 0 and $2\pi\frac{N-1}{N}$ on the unit circle in
the complex plane. The function $e^{j2\pi kn/N}=w^{kn}$ encircles the
origin $n\frac{N-1}{N}$ times, thus generating harmonics of the fundamental
sinusoid for which $n=1$.

The way in which we defined the DFT leads to a few subtleties
when $n>\frac{N}{2}$. The function $e^{j2\pi kn/N}$ is plotted
for increasing values of $k$ in Fig. ([fig:wkn values])
for the cases $n=1$ and $n=N-1$ for $N=16$. When $k$ increases from
$k$ to $k+1$, the angle increases by $\frac{2\pi n}{N}$. When
$n=1$, the step is $\frac{2\pi}{N}$. When $n=N-1$, the angle
increases by $2\pi\frac{N-1}{N}=2\pi-\frac{2\pi}{N}$. Since $2\pi
is$precisely once around the circle, the step equates to
$-\frac{2\pi}{N}$, i.e. in the direction of a negative
frequency. The components up to $N/2$ represent *positive* frequency
components, those above $N/2$ up to $N-1$ represent *negative*
frequencies with frequency. The angle increment for the component
$N/2$ for $N$ even advances precisely halfway around the circle for
each increment in $k$ and can therefore be interpreted as either a
positive or a negative frequency. This component of the DFT represents
the Nyquist Frequency, i.e. half of the sampling frequency and is
useful to orientate oneself when looking at DFT graphics.

The FFT in turn is simply a special and highly efficient algorithm for
calculating the DFT. Whereas a straightforward calculation of the DFT
takes of the order of $N^{2}$ calculations to compute, the FFT
algorithm requires of the order $N\log N$ calculations. The FFT was
the key to the wide-spread use of the DFT in real-time applications
and was included in a list of the top 10 algorithms of the $20^{th}$
century by the IEEE journal Computing in Science & Engineering in the
year 2000.

![[fig:wkn values]Unit circle samples](../figures/Unit circle samples.png)

Let’s apply the FFT to our radar data and see what happens!

### Signal properties in the frequency domain

First, we take the FFTs of our three signals and then display the
positive frequency components (i.e., components 0 to $N/2$).  These
are called the *range traces* in radar terminology.

```python
fig, axes = plt.subplots(3, 1, figsize=(15, 7))

# Take FFTs of our signals.  Note the convention to
# name FFTs with a capital letter.
V0 = np.fft.fft(v0)
V5 = np.fft.fft(v5)
V_actual = np.fft.fft(v_actual)

N = len(V0)

axes[0].plot(np.abs(V0[:N // 2]))
axes[0].set_ylabel("$|V_0|$")
axes[0].set_ylim(0,1100)

axes[1].plot(np.abs(V5[:N // 2]))
axes[1].set_ylabel("$|V_5 |$")
axes[1].set_ylim(0, 1000)

axes[2].plot(np.abs(V_actual[:N // 2]))
axes[2].set_ylim(0, 750)
axes[2].set_ylabel("$|V_\mathrm{actual}|$")

axes[2].set_xlabel("FFT component $n$")

for ax in axes:
    ax.grid()

plt.show()
```

Suddenly, the information makes sense!

Fig ([fig:FFT range traces]) shows the absolute values of the positive
frequency components (i.e. components 0 to $N/2$) of the FFTs of the
three signals in Fig. ([fig:radar time signals]), called *range
traces* in radar terminology. Suddenly the information makes sense!

The plot for $|V_{0}|$ clearly shows a target at component 67, that
for $|V5|$ shows the targets that produced the signal that was
uninterpretable in the time domain. The real radar signal shows a
large number of targets between component 400 and 500 with a large
peak in component 443. This happens to be an echo return from a radar
illuminating the high wall of an open-cast mine.

To get useful information from the plot, we must determine the range!
The sinusoid associated with the first component of the DFT has a
period exactly equal to the duration $T_{eff}$ of the time domain
signal, so $f_{1}=\frac{1}{T_{eff}}$. The other sinusoids in the
Fourier series are harmonics of this, $f_{n}=\frac{n}{T_{eff}}$.

The ranges associated with the DFT components follow from
Eqs. ([eq:difference frequency]) and ([eq:Effective bandwidth]) as

$$R_{n}=\frac{nv}{2B_{eff}}\label{eq:FFT ranges}$$

and the associated DFT components are known as *range bins* in radar
terminology.

This equation also defines the range resolution of the radar: targets
will only be distinguishable if they are separated by more than two
range bins, i.e.

$$\Delta R>\frac{1}{B_{eff}}.\label{eq:Range resolution}$$

This is a fundamental property of all types of radar.

The plot in Fig. ([fig:FFT range traces]) has a fundamental
shortcoming. The observable dynamic range is the signal is very
limited! We could easily have missed one of the targets in the trace
for $V_{5}$!  To ameliorate the problem, we plot the same FFTs but
this time against a logarithmic y-axis.  The traces were all
normalized by dividing the amplitudes with the maximum value.

```python
c = 3e8  # Approximately the speed of light and of
         # electro-magnetic waves in air

fig, (ax0, ax1,ax2) = plt.subplots(3, 1, sharex=True, figsize=(15, 7))


def dB(y):
    "Calculate the log ratio of y / max(y) in decibel."

    y = np.abs(y)
    y /= y.max()

    return 20 * np.log10(y)


def log_plot_normalized(x, y, ylabel, ax):
    ax.plot(x, dB(y))
    ax.set_ylabel(ylabel)
    ax.grid()


rng = np.arange(N // 2) * c / 2 / Beff

log_plot_normalized(rng, V0[:N // 2], "$|V_0|$ [dB]", ax0)
log_plot_normalized(rng, V5[:N // 2], "$|V_5|$ [dB]", ax1)
log_plot_normalized(rng, V_actual[:N // 2], "$|V_{\mathrm{actual}}|$ [dB]", ax2)

ax0.set_xlim(0, 300)  # Change x limits for these plots so that
ax1.set_xlim(0, 300)  # we are better able to see the shape of the peaks.

plt.show()
```

The observable dynamic range is much improved in these plots. For
instance, in the real radar signal the *noise floor* of the radar has
become visible. The noise floor is ultimately caused by a phenomenon
called thermal noise that is produced by all conducting elements that
have resistance at temperatures above absolute zero, as well as by
shot noise, a noise mechanism inherent in all the electronic devices
that are used for processing the radar signal. The noise floor of a
radar limits its ability to detect weak echoes.

### Windowing

The range traces in Fig. ([fig:Log range traces]) display a further
serious shortcoming. The signals $v_{1}$ and $v_{5}$ are composed of
pure sinusoids and we would ideally expect the FFT to produce line
spectra for these signals. The logarithmic plots show steadily
increasing values as the peaks are approached from both sides, to such
an extent that one of the targets in the plot for $|v_{5}|$ can hardly
be distinguished even though it is separated by several range bins
from the large target. The broadening is caused by *side lobes* in the
DFT. These are caused by an inherent clash between the properties of
the signal we analyzed and the signal produced by the inverse DFT.

![[fig:Periodicity anomaly]Eight samples have been taken of a given
 function with effective length $T_{eff}.$ A step discontinuity
 develops if the sampled function within the box is made periodic with
 period $T_{eff}.$ These discontinuities cause sidelobes in the
 sampled signal since the given periodic signal is not band-limited as
 required by the DFT. ](../figures/periodic.png)

Consider the function $x(t)$ shown in Figure . It is sampled at a
sampling frequency $f_{s}$ and $N=8$ samples are taken. The effective
length of the sampled signal is $T_{eff}=\frac{N}{f_{s}}$ as shown
in the figure. The time domain reconstruction of the signal $x_{n}$
according to equation ([eq:Inverse DFT]) is a periodic function with
period equal to $T_{eff}$ . Replicas of the sampled part of the signal
are shown towards the left and right of the original signal. There
clearly is a discrepancy in the form of step discontinuities at both
ends of the sampled function. A function with step discontinuities is
not a band-limited function, while the DFT requires a band-limited
function to faithfully reproduce the original time function. The step
discontinuities cause a broadening of the signal spectrum by the
formation of side lobes.

We can counter this effect by a process called *windowing*. The
original function is multiplied with a window function such as the
Kaiser window $K(N,\beta)$:

```python
f, axes = plt.subplots(1, 3, figsize=(10, 5))

for i, beta in enumerate([0, 3, 10]):
    axes[i].plot(np.kaiser(N, beta))
    axes[i].set_xlabel(r'$\beta = {}$'.format(beta))
    axes[i].set_xlim(0, N - 1)

plt.show()
```

By changing the parameter $\beta$, the shape of the
window can be changed from rectangular (i.e. no windowing) with
$\beta=0$ to a window that produces signals that smoothly increase
from zero and decrease to zero at the endpoints of the sampled
interval, producing very low side lobes with values of $\beta$ between
5 and 10.

Let's take a look at the signals used thus far in this example,
windowed with a Kaiser window with $\beta=6.1$:


```python
f, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 5))

t_ms = t * 1000  # Sample times in milli-second

w = np.kaiser(N, 6.1)  # Kaiser window with beta = 6.1

for n, (signal, label) in enumerate([(v0, r'$v_0 [Volt]$'),
                                     (v5, r'$v_5 [Volt]$'),
                                     (v_actual, r'$v_{\mathrm{actual}}$')]):
    axes[n].plot(t_ms, w * signal)
    axes[n].set_ylabel(label)
    axes[n].grid()

axes[2].set_xlim(0, t_ms[-1])
axes[2].set_xlabel('Time [ms]')
plt.show()

```

And the corresponding FFTs (or "range traces", in radar terms):

```python
V0_win = np.fft.fft(w * v0)
V5_win = np.fft.fft(w * v5)
V_actual_win = np.fft.fft(w * v_actual)

fig, (ax0, ax1,ax2) = plt.subplots(3, 1, figsize=(15, 7))

log_plot_normalized(rng, V0_win[:N // 2], r"$|V_0,\mathrm{win}|$ [dB]", ax0)
log_plot_normalized(rng, V5_win[:N // 2], r"$|V_5,\mathrm{win}|$ [dB]", ax1)
log_plot_normalized(rng, V_actual_win[:N // 2], r"$|V_\mathrm{actual,win}|$ [dB]", ax2)

ax0.set_xlim(0, 300)  # Change x limits for these plots so that
ax1.set_xlim(0, 300)  # we are better able to see the shape of the peaks.

ax1.annotate("New, previously unseen!", (160, -35),
             xytext=(10, 25), textcoords="offset points", color='red',
             arrowprops=dict(width=2, headwidth=6, frac=0.4, shrink=0.1))

plt.show()

```

Compare these with the range traces in Figure
[fig:Log range traces]. There is a dramatic lowering in side lobe
level, but this came at a price: the peaks have changed in shape,
widening and becoming less peaky, thus lowering the radar resolution,
that is, the ability of the radar to distinguish between two closely
space targets. The choice of window is a compromise between side lobe
level and resolution. Even so, referring to the trace for $V_{5}$,
windowing has dramatically increased our ability to distinguish the
small target from its large neighbor.

In the real radar data range trace windowing has also reduced the side
lobes. This is most visible in the depth of the notch between the two
groups of targets.

### Radar Images

With the key concepts sorted out, we can now look at radar images.

The data is produced by a radar with a parabolic reflector antenna. It
produces a highly directive round pencil beam with a $2^\circ$
spreading angle between half-power points. When directed with normal
incidence at a plane, the radar will illuminate a spot of about 2 m in
diameter on the half power contour at a distance of 60 m. Outside this
spot the power drops off quite rapidly but strong echoes from outside
the spot will nevertheless still be visible.

A rock slope consists of thousands of scatterers. A range bin can be
thought of as a large sphere with the radar at its center that
intersects the slope along a ragged line. The scatterers on this line
will produce reflections in this range bin. The scatterers are
essentially randomly arranged along the line. The wavelength of the
radar is about 30 mm. The reflections from scatterers separated by odd
multiples of a quarter wavelength in range, about 7.5 mm, will tend to
interfere destructively, while those from scatterers separated by
multiples of a half wavelength will tend to interfere constructively
at the radar. The reflections combine to produce apparent spots of
strong reflections. Radar measurements of a small scanned region
consisting of 20 azimuth and 30 elevation bins scanned in steps of
$0.5^\circ$.


```python
data = np.load('data/radar_scan_1.npz')
scan = data['scan']

# ADC is 14-bit for 5V max, so scale by 0.000305 to return to volt
v = scan['samples'] * 0.000305

# Take FFT for each measurement
V = np.fft.fft(v, axis=2)[::-1, :, :N // 2]

contours = np.arange(-40, 1, 2)

f, axes = plt.subplots(1, 3, figsize=(16, 5))

for n, (radar_slice, title) in enumerate([ (V[:, :, 250], 'Range Sphere 250'),
                                           (V[6, :, :], 'Elevation Plane 6'),
                                           (V[:,  3, :], 'Azimuth Plane 3') ]):

    axes[n].contourf(dB(radar_slice), contours)
    axes[n].set_title(title)

plt.show()

```


Contour plots of the radar data, showing the strength of echoes
against elevation and azimuth, a cut through the slope in an elevation
plane and acut through the slope in an azimuth plane. The contours are
in steps of 2 dB from -40 to 0 dB. Azimuth and elevation bin size is
$0.5^\circ$ and range bin size is 1.5 m. The stepped construction of
the high wall in an opencast mine is clearly visible.

### Further applications of the FFT in radar

The examples above show just one of the uses of the FFT in radar. The
FFT provides us with a versatile tool that finds many other uses in
radar, including pulse expansion and compression, Doppler measurement
and detection, one and two dimensional beam forming in antennas and
target recognition.
