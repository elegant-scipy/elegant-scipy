# Function optimization in SciPy

We are going to use SciPy's `optimize` module to align two images, using the
method described in the papers:

* Pluim et al., Image registration by maximization of combined mutual
  information and gradient information, IEEE Transactions on Medical
  Imaging, 19(8) 2000

and

* Pluim et al., Mutual-Information-Based Registration of Medical
  Images: A Survey, IEEE Transactions on Medical Imaging, 22(8) 2003

We start, of course, by setting up our plotting environment:

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('style/elegant.mplstyle')
```

Correlation-based image alignment depends on detecting the similarity between
two images at various levels of alignment. We can then "jiggle" the images, and
see whether jiggling them in one direction or another improves the similarity.
By doing this repeatedly, we can try to find the correct alignment.

There are a few problems with the approach as we've just described it, but
we'll deal with the first one: how do you define "image similarity"? One metric
is the *normalized mutual information*, or NMI, which measures how easy it
would be to predict a pixel value of one image given the value of the
corresponding pixel in the other. See chapter 5 for some more information on
this, as well as the original reference:

* Studholme, C., Hill, D.L.G., Hawkes, D.J.: An Overlap Invariant Entropy Measure
  of 3D Medical Image Alignment. Patt. Rec. 32, 71–86 (1999)

Let's start with the definition of the mutual information:

```python
import numpy as np

from scipy.stats import entropy
from scipy import optimize
from scipy import ndimage as ndi

from skimage import transform
from skimage.util import random_noise


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

With this, we can already check whether two images are aligned:

```python
# TODO: show NMI values for images perfectly aligned, slightly off, and very
# off alignment
```

This is the crux of optimization: define a function to optimize (in this case,
the NMI of two images for a given offset value), optimize it, and apply the
resulting parameter set (in this case, align the images with the found offset):

```python
# TODO: start with a ludicrously close offset, so that there's no local minima,
# then apply scipy.optimize to find the correct offset (0)
```

Unfortunately, this brings us to the principal difficulty of this kind of
alignment: sometimes, the NMI has to get worse before it gets better. Have a
look at the NMI value as we slide the same image past itself. The perfect
alignment, of course, is an offset of 0:

```python
# TODO: plot NMI for many offset values along the column axis.
```

As you can see, if you start out with an offset value of , when you jiggle the
offset, the NMI will always get worse. You might then erroneously conclude that
you are done with the alignment, when in fact you are quite a ways off.

The common solution to this problem is to smooth or downscale the images, which
has the dual result of smoothing the objective function. Have a look at the
same plot, after having smoothed the images with a Gaussian filter:

```python
# TODO: smooth image and repeat above plot
```

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

    w = (np.cos(2 * alpha) + 1) / 2

    w[np.isclose(mag_g_A, 0)] = 0
    w[np.isclose(mag_g_B, 0)] = 0

    return w * np.minimum(mag_g_A, mag_g_B)


def alignment(A, B, sigma=1.5, normalized=True):
    I = mutual_information(A, B, normalized=normalized)
    G = np.sum(gradient_similarity(A, B, sigma=sigma))

    return I * G


def build_tf(p):
    r, tx, ty = p
    return transform.SimilarityTransform(rotation=r,
                                         translation=(tx, ty))


def cost(p, X, Y):
    tf = build_tf(p)
    Y_prime = transform.warp(Y, tf, order=3)

    return -1 * alignment(X, Y_prime)


# Generalize this for N-d
def register(A, B):
    pyramid_A = tuple(transform.pyramid_gaussian(A, downscale=2))
    pyramid_B = tuple(transform.pyramid_gaussian(B, downscale=2))
    N = range(len(pyramid_A))
    image_pairs = zip(N, pyramid_A, pyramid_B)

    p = np.array([0, 0, 0])

    for (n, X, Y) in reversed(list(image_pairs)):
        if X.shape[0] < 5:
            continue

        print('   .  ')
        print('  / \ Pyramid scaled down by 2x {}'.format(n))
        print(' /   \ ')
        print('.-----.')

        p[1:] *= 2

        res = optimize.minimize(cost,
                                p,
                                args=(X, Y),
                                method='Powell')
        p = res.x

        print('Angle:', np.rad2deg(res.x[0]))
        print('Offset:', res.x[1:] * 2 ** n)
        print('Cost function:', res.fun)
        print('')

    return build_tf(p)
```

Next, we use those functions to align two parts of an image:

```python
from skimage import data, transform, color

img0 = transform.rescale(color.rgb2gray(data.astronaut()), 0.3)

theta = 40
img1 = transform.rotate(img0, theta)
img1 = random_noise(img1, mode='gaussian', seed=0, mean=0, var=1e-3)

tf = register(img0, img1)
corrected = transform.warp(img1, tf, order=3)


f, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(img0, cmap='gray')
ax0.set_title('Input image')
ax1.imshow(img1, cmap='gray')
ax1.set_title('Transformed image + noise')
ax2.imshow(corrected, cmap='gray')
ax2.set_title('Registered image')

print('Calculating cost function profile...')
f, ax0 = plt.subplots()
angles = np.linspace(-theta - 20, -theta + 20, 51)
costs = [-1 * alignment(img0, transform.rotate(img1, angle)) for angle in angles]
ax0.plot(angles, costs)
ax0.set_title('Cost function around angle of interest')
ax0.set_xlabel('Angle')
ax0.set_ylabel('Cost')

plt.show()
```
