# Function optimization in SciPy

> Life is like a landscape. You live in the midst of it but can
> describe it only from the vantage point of distance. --- Charles
> Lindbergh

Hanging a picture on the wall, it is sometimes hard to get it
straight.  You make an adjustment, step back, evaluate the picture's
horizontality, and repeat.  This is a process of *optimization*: we're
changing the orientation of the portrait until it satisfies our
demand---that it makes a zero angle with the horizon.

In mathematics, our demand is called a "cost function", and the
orientation of the portrait the "parameter".  In a typical
optimization problem, we vary the parameters until the cost function
is minimized.

Consider, for example, the shifted parabola, $f(x) = (x - 3)^2$.  We
know that this function, with parameter $x$, has a minimum at 3,
because we can calculate the derivative, set it to zero, and see that
$2 (x - 3) = 0$, i.e. $x = 3$.

But, if this function were much more complicated (e.g., was an
expression with many terms, had multiple points of zero derivative,
contained non-linearities, or was dependent on more variables), using
a hand calculation would become arduous.

You can think of the cost function as representing a landscape, where we
are trying to find the lowest point.  That analogy immediately
highlights one of the hard parts of this problem: if you are standing
in any valley, with mountains surrounding you, how do you know whether
you are in the lowest valley, or whether the valley is perhaps
surrounded by *even taller* mountains?  In optimization parlance: how
do you know whether you are trapped in a *local
minimum*?  Most of the optimization algorithms available make some
attempt to address the issue[^line_search].

[^line_search]: Optimization algorithms handle this issue in various
                ways, but two common approaches are line searches and
                trust regions.  With a *line search*, you try to find
                the cost function minimum along a specific dimension,
                and then successively attempt the same along the other
                dimensions.  With *trust regions*, we move our guess
                for the minimum in the direction we expect it to be;
                if we see that we are indeed approaching the minimum
                as expected, we repeat the procedure with increased
                confidence.  If not, we lower our confidence and
                search a wider area.

![Optimization Comparison](../figures/generated/optimization_comparison.png)

There are many different optimization algorithms to choose from (see
figure).  You get to choose whether your cost function takes a scalar
or a vector as input (i.e., do you have one or multiple parameters to
optimize?).  There are those that require the cost function gradient
to be given and those that automatically estimate it.  Some only
search for parameters in a given area (*constrained optimization*),
and others examine the entire parameter space.

In the rest of this chapter, we are going to use SciPy's `optimize`
module to align two images, using the method described in the papers:

* Pluim et al., Image registration by maximization of combined mutual
  information and gradient information, IEEE Transactions on Medical
  Imaging, 19(8) 2000

and

* Pluim et al., Mutual-Information-Based Registration of Medical
  Images: A Survey, IEEE Transactions on Medical Imaging, 22(8) 2003

Applications of image alignment or *registration* include panorama
stitching, combination of multi-modal brain scans, super-resolution
imaging, and, in astronomy, object denoising through the combination
of multiple exposures.

We start, now, by setting up our plotting environment:

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('style/elegant.mplstyle')
```

Let's start with the simplest version of the problem: we have two
images, one shifted relative to the other.  We wish to recover the
shift that will best align our images.

Our optimization function will "jiggle" one of the images, and see
whether jiggling it in one direction or another reduces their
dissimilarity.  By doing this repeatedly, we can try to find the
correct alignment.

For the optimization algorithm to do its work, we need some way of
defining "dissimilarity"---i.e., the cost function.  The easiest is to
simply calculate the sum of the squared differences:

```python
import numpy as np

def ssd(A, B):
    """Sum of squared differences."""
    return np.sum((A - B)**2) / A.size
```

This will return 0 when the images are perfectly aligned, and a higher
value otherwise.

With this cost function, we can check whether two images are aligned:

```python
from scipy import ndimage as ndi
from skimage import data, color

astronaut = color.rgb2gray(data.astronaut())
ncol = astronaut.shape[1]

shifts = np.linspace(-0.9 * ncol, 0.9 * ncol, 181)
ssd_costs = []

for shift in shifts:
    shifted = ndi.shift(astronaut, (0, shift))
    ssd_costs.append(ssd(astronaut, shifted))

fig, ax = plt.subplots()
ax.plot(shifts, ssd_costs)
```

With the cost function defined, we can ask `scipy.optimize.minimize`
to search for optimal parameters:

```python
from scipy import optimize

shifted1 = ndi.shift(astronaut, (0, 50))

def astronaut_shift_error(shift, image):
    corrected = ndi.shift(image, (0, shift))
    return ssd(astronaut, corrected)

res = optimize.minimize(astronaut_shift_error, 0, args=(shifted1,),
                        method='Powell')

print('The optimal shift for correction is: %f' % res.x)
```

Brilliant! Thanks to our SSD measure, SciPy's `optimize.minimize` function has
recovered the correct amount to shift our distorted image to get it back to its
original state.

Unfortunately, this brings us to the principal difficulty of this kind of
alignment: sometimes, the SSD has to get worse before it gets better. Have a
look at the SSD value as the shift gets larger and larger: at around -300
pixels of shift, it starts to decrease again! Only slightly, but it decreases
nonetheless. Because optimization methods only have access to "nearby"
values of the cost function, if the function improves by moving in the "wrong"
direction, the `minimize` process will move that way regardless. So, if we
start by an image shifted by -340 pixels:

```python
shifted2 = ndi.shift(astronaut, (0, -340))
```

`minimize` will shift it by a further 40 pixels or so, instead of recovering
the original image:

```python
res = optimize.minimize(astronaut_shift_error, 0, args=(shifted2,),
                        method='Powell')

print('The optimal shift for correction is %f' % res.x)
```

The common solution to this problem is to smooth or downscale the images, which
has the dual result of smoothing the objective function. Have a look at the
same plot, after having smoothed the images with a Gaussian filter:

```python
from skimage import filters

astronaut_smooth = filters.gaussian(astronaut, sigma=20)

ssd_costs_smooth = []
shifts = np.linspace(-0.9 * ncol, 0.9 * ncol, 181)
for shift in shifts:
    shifted = ndi.shift(astronaut_smooth, (0, shift))
    ssd_costs_smooth.append(ssd(astronaut_smooth, shifted))

fig, ax = plt.subplots()
ax.plot(shifts, ssd_costs, label='original')
ax.plot(shifts, ssd_costs_smooth, label='smoothed')
ax.legend(loc='lower right')
ax.set_xlabel('Shift')
ax.set_ylabel('SSD')
```

As you can see, with some rather extreme smoothing, the "funnel" of
the error function becomes wider, and less bumpy. Therefore, modern alignment
software uses what's called a *Gaussian pyramid*, which is a set of
progressively lower resolution versions of the same image.  We align
the the lower resolution (blurrier) images first, to get an
approximate alignment, and then progressively refine the alignment
with sharper images.

```python

def downsample2x(image):
    offsets = [((s + 1) % 2) / 2 for s in image.shape]
    slices = [slice(offset, end, 2)
              for offset, end in zip(offsets, image.shape)]
    coords = np.mgrid[slices]
    return ndi.map_coordinates(image, coords, order=1)


def gaussian_pyramid(image, levels=6):
    """Make a Gaussian image pyramid.

    Parameters
    ----------
    image : array of float
        The input image.
    max_layer : int, optional
        The number of levels in the pyramid.

    Returns
    -------
    pyramid : iterator of array of float
        An iterator of Gaussian pyramid levels, starting with the top
        (lowest resolution) level.
    """
    pyramid = [image]

    for level in range(levels - 1):
        blurred = ndi.gaussian_filter(image, sigma=2/3)
        image = downsample2x(image)
        pyramid.append(image)

    return reversed(pyramid)
```

Let's see how the 1D alignment looks along that pyramid:

```python
shifts = np.linspace(-0.9 * ncol, 0.9 * ncol, 181)
nlevels = 6
costs = np.empty((nlevels, len(shifts)), dtype=float)
astronaut_pyramid = list(gaussian_pyramid(astronaut, levels=nlevels))
for col, shift in enumerate(shifts):
    shifted = ndi.shift(astronaut, (0, shift))
    shifted_pyramid = gaussian_pyramid(shifted, levels=nlevels)
    for row, image in enumerate(shifted_pyramid):
        costs[row, col] = ssd(astronaut_pyramid[row], image)

fig, ax = plt.subplots()
for level, cost in enumerate(costs):
    ax.plot(shifts, cost, label='Level %d' % (nlevels - level))
ax.legend(loc='lower right', frameon=True, framealpha=0.9)
ax.set_xlabel('Shift')
ax.set_ylabel('SSD')
```

As you can see, at the highest level of the pyramid, that bump at a shift of
about -325 disappears. We can therefore get an approximate alignment at that
level, then pop down to the lower levels to refine that alignment.

Let's automate that, and try with a "real" alignment, with three parameters:
rotation, translation in the row dimension, and translation in the
column dimension. (This is called a "*rigid* registration".) To simplify the
code, we'll use the scikit-image *transform* module to compute the shift and
rotation of the image.

```python
from skimage import transform

def build_tf(param):
    r, tx, ty = param
    return transform.SimilarityTransform(rotation=r,
                                         translation=(tx, ty))


def cost_ssd(param, X, Y):
    transformation = build_tf(param)
    Y_prime = transform.warp(Y, transformation, order=3)
    return ssd(X, Y_prime)


# TODO: Generalize this for N-d
def align(A, B, cost=cost_ssd):
    nlevels = 7
    pyramid_A = gaussian_pyramid(A, levels=nlevels)
    pyramid_B = gaussian_pyramid(B, levels=nlevels)

    levels = range(nlevels, -1, -1)
    image_pairs = zip(pyramid_A, pyramid_B)

    p = np.zeros(3)

    for n, (X, Y) in zip(levels, image_pairs):
        p[1:] *= 2

        res = optimize.minimize(cost, p, args=(X, Y), method='Powell')
        p = res.x

        print('Pyramid level %i' % n)
        print('Angle:', np.rad2deg(res.x[0]))
        print('Offset:', res.x[1:] * 2 ** n)
        print('Cost function:', res.fun)
        print('')

    return build_tf(p)
```

```python
from skimage import data, transform, color, util

img0 = color.rgb2gray(data.astronaut())
theta = 60
img1 = transform.rotate(img0, theta)
img1 = util.random_noise(img1, mode='gaussian', seed=0, mean=0, var=1e-3)

tf = align(img0, img1)
corrected = transform.warp(img1, tf, order=3)


f, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(img0, cmap='gray')
ax0.set_title('Input image')
ax1.imshow(img1, cmap='gray')
ax1.set_title('Transformed image + noise')
ax2.imshow(corrected, cmap='gray')
ax2.set_title('Registered image')
```

We're feeling pretty good now.  And then a friend from neuroimaging
challenges us to use our new method to align two brain volumes, one
from a PET scan, the other from an MRI scan.

Thinking about it, the Sum of Squared Differences no longer seems to
be such a good idea.  PET and MRI images look very dissimilar!

Let's examine, for example, how the SSD cost function would vary with
increasing angles of rotation:

```python
# MODIFY WITH BRAIN IMAGES
# POSSIBLY FROM http://www.insight-journal.org/rire/ ?

f, ax0 = plt.subplots()
pyr0 = transform.pyramid_gaussian(img0, downscale=2, max_layer=5)
pyr1 = transform.pyramid_gaussian(img1, downscale=2, max_layer=5)
image_pairs = list(zip(pyr0, pyr1))
n_levels = len(image_pairs)
angles = np.linspace(-theta - 180, -theta + 180, 201)
for n, (X, Y) in zip(range(n_levels, 0, -1),
                     reversed(list(image_pairs))):
    costs = np.array([ssd(X, transform.rotate(Y, angle))
                      for angle in angles])
    costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))
    ax0.plot(angles, costs, label='level %i' % n)
ax0.set_title('Cost function around angle of interest')
ax0.set_xlabel('Angle')
ax0.set_ylabel('Cost')
ax0.legend(loc='lower right', frameon=True, framealpha=0.9)

plt.show()
```

Since SSD won't work, we have to search for a better cost function.
It's not altogether surprising that cost functions tend to be highly domain and
problem specific!

A suggested metric for multi-modal images, like the brain scans here,
is  *normalized mutual information*, or NMI, which measures how easy it
would be to predict a pixel value of one image given the value of the
corresponding pixel in the other[^nmi_paper].

[^nmi_paper]: This measure was defined in the paper: Studholme, C.,
              Hill, D.L.G., Hawkes, D.J.: An Overlap Invariant
              Entropy Measure of 3D Medical Image
              Alignment. Patt. Rec. 32, 71–86 (1999)

Let's start with the definition of NMI:

$$
I(X, Y) = \frac{H(X) + H(Y)}{H(X, Y)},
$$

where $H(X)$ is the *entropy* of $X$, and $H(X, Y)$ is the joint
entropy of $X$ and $Y$.  The numerator describes the entropy of the
two images, seen separately, and the denomenator the total entropy if
they are observed together.  Values can vary between 1 (maximally
aligned) and 2 (minimally aligned)[^mi_calc]. (See Chapter 5 for a
more in-depth discussion of entropy.)

[^mi_calc]: A quick handwavy explanation is that entropy is calculated
            from the histogram of the quantity under consideration.
            If $X = Y$, then the joint histogram $(X, Y)$ is diagonal,
            and that diagonal is the same as that of either $X$ or
            $Y$.  Thus, $H(X) = H(Y) = H(X, Y)$ and $I(X, Y) = 2$.

We compute normalized mutual information as follows:

```python
from scipy.stats import entropy

def normalized_mutual_information(A, B):
    """Compute the normalized mutual information.

    The normalized mutual information is given by:

                H(A) + H(B)
      Y(A, B) = -----------
                  H(A, B)

    where H(X) is the entropy ``- sum(x log x) for x in X``.

    Parameters
    ----------
    A, B : ndarray
        Images to be registered.

    Returns
    -------
    nmi : float
        The normalized mutual information between the two arrays, computed at a
        granularity of 100 bins per axis (10,000 bins total).
    """
    hist, bin_edges = np.histogramdd([np.ravel(A), np.ravel(B)], bins=100)
    hist /= np.sum(hist)

    H_A = entropy(np.sum(hist, axis=0))
    H_B = entropy(np.sum(hist, axis=1))
    H_AB = entropy(np.ravel(hist))

    return (H_A + H_B) / H_AB
```

Now, let's attempt the same optimization as before, this time using
mutual information:

<!-- ```python -->
<!-- import numpy as np -->

<!-- from scipy import optimize -->
<!-- from scipy import ndimage as ndi -->

<!-- from skimage import transform -->
<!-- from skimage.util import random_noise -->


```python
# Add example here
```

In difficult cases, NMI registration may still fail.  E.g.:

```python
# register two tricky images
```

<!-- Thus we can see that the mutual information fails dramatically at the -->
<!-- higher (low-resolution) pyramid levels, while it is virtually flat -->
<!-- at the lower (high-resolution) levels, except for a narrow region around the true -->
<!-- transformation angle! If you start with a relatively faraway angle, -->
<!-- you have no hope of recovering the true transformation. -->

In their 2000 paper, Pluim *et al.* showed that you can improve the properties
of the objective function by adding *gradient* information to the NMI metric.
Let's try it here to see if we can improve our alignment:

```python
def gradient(image, sigma=1):
    gaussian_filtered = ndi.gaussian_filter(image, sigma=sigma,
                                            mode='constant', cval=0)
    return np.gradient(gaussian_filtered)


def gradient_norm(g):
    return np.linalg.norm(g, axis=-1)


def gradient_similarity(A, B, sigma=1, scale=True):
    """For each pixel, calculate the angle between the gradients of A & B.

    Parameters
    ----------
    A, B : ndarray
        Images.
    sigma : float
        Sigma for the Gaussian filter, used to calculate the image gradient.

    Notes
    -----
    In multi-modal images, gradients may often be similar but point
    in opposite directions.  This weighting function compensates for
    that by mapping both 0 and pi to 1.

    Different imaging modalities can highlight different structures.  We
    are only interested in edges that occur in both images, so we scale the
    similarity by the minimum of the two gradients.

    """
    g_A = np.dstack(gradient(A, sigma=sigma))
    g_B = np.dstack(gradient(B, sigma=sigma))

    mag_g_A = gradient_norm(g_A)
    mag_g_B = gradient_norm(g_B)

    alpha = np.arccos(np.sum(g_A * g_B, axis=-1) /
                        (mag_g_A * mag_g_B))

    # Transform the alpha so that gradients that point in opposite directions
    # count as the same angle
    w = (np.cos(2 * alpha) + 1) / 2

    w[np.isclose(mag_g_A, 0)] = 0
    w[np.isclose(mag_g_B, 0)] = 0

    return w * np.minimum(mag_g_A, mag_g_B)


def alignment(A, B, sigma=1.5):
    I = normalized_mutual_information(A, B)
    G = np.sum(gradient_similarity(A, B, sigma=sigma))

    return I * G


```

We attempt the registration again:

```python
def cost_alignment(param, X, Y):
    transformation = build_tf(param)
    Y_prime = transform.warp(Y, transformation, order=3)
    return -alignment(X, Y_prime)


tf = align(img0, img1, cost=cost_alignment)
corrected = transform.warp(img1, tf, order=3)


f, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(img0, cmap='gray')
ax0.set_title('Input image')
ax1.imshow(img1, cmap='gray')
ax1.set_title('Transformed image + noise')
ax2.imshow(corrected, cmap='gray')
ax2.set_title('Registered image')
```

Success! Let's look at the objective function profile:

```python
print('Calculating cost function profile...')
f, ax0 = plt.subplots()
pyr0 = transform.pyramid_gaussian(img0, downscale=2, max_layer=5)
pyr1 = transform.pyramid_gaussian(img1, downscale=2, max_layer=5)
image_pairs = list(zip(pyr0, pyr1))
n_levels = len(image_pairs)
angles = np.linspace(-theta - 180, -theta + 180, 201)
for n, (X, Y) in zip(range(n_levels, 0, -1),
                     reversed(list(image_pairs))):
    costs = np.array([-alignment(X, transform.rotate(Y, angle))
                      for angle in angles])
    costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))
    ax0.plot(angles, costs, label='level %i' % n)
ax0.set_title('Cost function around angle of interest')
ax0.set_xlabel('Angle')
ax0.set_ylabel('Cost')
ax0.legend(loc='lower right')

plt.show()
```

As you can see, it's much more funnel-shaped than NMI by itself. It's
easy to see why progressive optimization of Pluim's function at
decreasing levels of the pyramid would result in the correct
alignment.

