# Function optimization in SciPy

> "What's new?" is an interesting and broadening eternal question, but one
> which, if pursued exclusively, results only in an endless parade of trivia
> and fashion, the silt of tomorrow. I would like, instead, to be concerned
> with the question "What is best?", a question which cuts deeply rather than
> broadly, a question whose answers tend to move the silt downstream.
>
> — Robert M Pirsig, Zen and the Art of Motorcycle Maintenance

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

We start, as usual, by setting up our plotting environment:

```python
# Make plots appear inline, set custom plotting style
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
simply calculate the average of the squared differences, often called the
*mean squared error*, or MSE. As in previous chapters, images will just be
NumPy arrays.

```python
import numpy as np

def mse(arr1, arr2):
    """Compute the mean squared error between two arrays."""
    return np.mean((arr1 - arr2)**2)
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
mse_costs = []

for shift in shifts:
    shifted = ndi.shift(astronaut, (0, shift))
    mse_costs.append(mse(astronaut, shifted))

fig, ax = plt.subplots()
ax.plot(shifts, mse_costs)
```

With the cost function defined, we can ask `scipy.optimize.minimize`
to search for optimal parameters:

```python
from scipy import optimize

shifted1 = ndi.shift(astronaut, (0, 50))

def astronaut_shift_error(shift, image):
    corrected = ndi.shift(image, (0, shift))
    return mse(astronaut, corrected)

res = optimize.minimize(astronaut_shift_error, 0, args=(shifted1,),
                        method='Powell')

print('The optimal shift for correction is: %f' % res.x)
```

Brilliant! Thanks to our MSE measure, SciPy's `optimize.minimize` function has
recovered the correct amount to shift our distorted image to get it back to its
original state.

Unfortunately, this brings us to the principal difficulty of this kind of
alignment: sometimes, the MSE has to get worse before it gets better. Have a
look at the MSE value as the shift gets larger and larger: at around -300
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

mse_costs_smooth = []
shifts = np.linspace(-0.9 * ncol, 0.9 * ncol, 181)
for shift in shifts:
    shifted = ndi.shift(astronaut_smooth, (0, shift))
    mse_costs_smooth.append(mse(astronaut_smooth, shifted))

fig, ax = plt.subplots()
ax.plot(shifts, mse_costs, label='original')
ax.plot(shifts, mse_costs_smooth, label='smoothed')
ax.legend(loc='lower right')
ax.set_xlabel('Shift')
ax.set_ylabel('MSE')
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
nlevels = 8
costs = np.empty((nlevels, len(shifts)), dtype=float)
astronaut_pyramid = list(gaussian_pyramid(astronaut, levels=nlevels))
for col, shift in enumerate(shifts):
    shifted = ndi.shift(astronaut, (0, shift))
    shifted_pyramid = gaussian_pyramid(shifted, levels=nlevels)
    for row, image in enumerate(shifted_pyramid):
        costs[row, col] = mse(astronaut_pyramid[row], image)

fig, ax = plt.subplots()
for level, cost in enumerate(costs):
    ax.plot(shifts, cost, label='Level %d' % (nlevels - level))
ax.legend(loc='lower right', frameon=True, framealpha=0.9)
ax.set_xlabel('Shift')
ax.set_ylabel('MSE')
```

As you can see, at the highest level of the pyramid, that bump at a shift of
about -325 disappears. We can therefore get an approximate alignment at that
level, then pop down to the lower levels to refine that alignment.

Let's automate that, and try with a "real" alignment, with three parameters:
rotation, translation in the row dimension, and translation in the
column dimension. (This is called a "*rigid* registration".)

To simplify the code, we'll use the scikit-image *transform* module to compute
the shift and rotation of the image. SciPy's `optimize` requires a vector of
parameters as input. These are just numbers, without meaning. We first make a
function that will take such a vector and produce a rigid transformation with
the right parameters:

```python
from skimage import transform

def make_rigid_transform(param):
    r, tc, tr = param
    return transform.SimilarityTransform(rotation=r,
                                         translation=(tc, tr))
```

Next, we need a cost function. This is just MSE, but SciPy requires a specific
format: the first argument needs to be the *parameter vector*, which it is
optimizing. Subsequent arguments can be passed through the `args` keyword as a
tuple, but must remain fixed: only the parameter vector can be optimized. In
our case, this is just the rotation angle and the two translation parameters:

```python
def cost_mse(param, reference_image, target_image):
    transformation = make_rigid_transform(param)
    transformed = transform.warp(target_image, transformation, order=3)
    return mse(reference_image, transformed)
```

Finally, we write our alignment function, which optimizes our cost function
*at each level of the Gaussian pyramid*, using the result of the previous
level as a starting point for the next one:

```python
def align(reference, target, cost=cost_mse):
    nlevels = 7
    pyramid_ref = gaussian_pyramid(reference, levels=nlevels)
    pyramid_tgt = gaussian_pyramid(target, levels=nlevels)

    levels = range(nlevels, 0, -1)
    image_pairs = zip(pyramid_ref, pyramid_tgt)

    p = np.zeros(3)

    for n, (ref, tgt) in zip(levels, image_pairs):
        p[1:] *= 2

        res = optimize.minimize(cost, p, args=(ref, tgt), method='Powell')
        p = res.x

        print('Pyramid level %i' % n)
        print('Angle:', np.rad2deg(res.x[0]))
        print('Offset:', res.x[1:] * 2 ** n)
        print('Cost function:', res.fun)
        print('')

    return make_rigid_transform(p)
```

Let's try it with our astronaut image. We rotate it by 60 degrees and add some
noise to it. Can SciPy recover the correct transform?

```python
from skimage import util

theta = 60
rotated = transform.rotate(astronaut, theta)
rotated = util.random_noise(rotated, mode='gaussian',
                            seed=0, mean=0, var=1e-3)

tf = align(astronaut, rotated)
corrected = transform.warp(rotated, tf, order=3)

f, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(astronaut)
ax0.set_title('Original')
ax1.imshow(rotated)
ax1.set_title('Rotated')
ax2.imshow(corrected)
ax2.set_title('Registered')
for ax in (ax0, ax1, ax2):
    ax.axis('off')
```

We're feeling pretty good now. But our choice of parameters actually masked
the difficulty of optimization: Let's see what happens with a rotation of
50 degrees, which is *closer* to the original image:

```python
theta = 50
rotated = transform.rotate(astronaut, theta)
rotated = util.random_noise(rotated, mode='gaussian',
                            seed=0, mean=0, var=1e-3)

tf = align(astronaut, rotated)
corrected = transform.warp(rotated, tf, order=3)

f, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(astronaut)
ax0.set_title('Original')
ax1.imshow(rotated)
ax1.set_title('Rotated')
ax2.imshow(corrected)
ax2.set_title('Registered')
for ax in (ax0, ax1, ax2):
    ax.axis('off')

```

Oops! Even though we started closer to the original image, we failed to
recover the correct rotation. This is because optimization techniques can get
stuck in local minima, little bumps on the road to success, as we saw above
with the shift-only alignment. They can therefore be quite sensitive to the
starting parameters.

<!-- exercise begin -->

**Exercise:** Try incorporating the `scipy.optimize.basinhopping` function
into the `align` function, which has explicit strategies to avoid local minima.

*Hint:* limit basinhopping to the top levels of the pyramid, as it is a slower
optimization approach, and could take rather long to run at full image
resolution.

*Note:* you need the solution to this exercise for subsequent alignments, so
look it up in the solutions if you are having trouble getting it to work.

<!-- solution begin -->

**Solution:** We use basin-hopping at the higher levels of the pyramid, but use
Powell's method for the lower levels, because basin-hopping is too
computationally expensive to run at full resolution:

```python
def align(reference, target, cost=cost_mse, nlevels=7, method='Powell'):
    pyramid_ref = gaussian_pyramid(reference, levels=nlevels)
    pyramid_tgt = gaussian_pyramid(target, levels=nlevels)

    levels = range(nlevels, 0, -1)
    image_pairs = zip(pyramid_ref, pyramid_tgt)

    p = np.zeros(3)

    for n, (ref, tgt) in zip(levels, image_pairs):
        p[1:] *= 2
        if method.upper() == 'BH':
            res = optimize.basinhopping(cost, p,
                                        minimizer_kwargs={'args': (ref, tgt)})
            if n <= 4:
                method = 'Powell'
        else:
            res = optimize.minimize(cost, p, args=(ref, tgt), method='Powell')
        p = res.x
        print('Pyramid level %i' % n)
        print('Angle:', np.rad2deg(res.x[0]))
        print('Offset:', res.x[1:] * 2 ** n)
        print('Cost function:', res.fun)
        print('')

    return make_rigid_transform(p)
```

Now let's try that alignment:

```python
from skimage import util

theta = 50
rotated = transform.rotate(astronaut, theta)
rotated = util.random_noise(rotated, mode='gaussian',
                            seed=0, mean=0, var=1e-3)

tf = align(astronaut, rotated, nlevels=8, method='BH')
corrected = transform.warp(rotated, tf, order=3)

f, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(astronaut)
ax0.set_title('Original')
ax1.imshow(rotated)
ax1.set_title('Rotated')
ax2.imshow(corrected)
ax2.set_title('Registered')
for ax in (ax0, ax1, ax2):
    ax.axis('off')
```

Boom! Consider that basin *hopped!*

<!-- solution end -->

<!-- exercise end -->

At this point, we have a working registration approach, which is most
excellent. But it turns out that we've only solved the
easiest of registration problems: images of the same *modality*. This means
that we expect bright pixels in the reference image to match up to bright
pixels in the test image. We now move on to aligning different color channels
of the same image. This task has historical significance: between 1909 and
1915, the photographer Sergei Mikhailovich Prokudin-Gorskii produced color
photographs of the Russian empire before color photography had been invented.
He did this by taking three different monochrome pictures of a scene, each
with a different color filter placed in front of the lens.

Aligning bright pixels together, as the MSE implicitly does, won't work in
this case. Take, for example, these three pictures of a stained glass window
in the Church of Saint John the Theologian, taken from the [Library of Congress
Prokudin-Gorskii Collection](http://www.loc.gov/pictures/item/prk2000000263/):

```python
from skimage import io
stained_glass = io.imread('data/00998v.jpg') / 255  # use float image in [0, 1]
fig, ax = plt.subplots()
ax.imshow(stained_glass)
ax.axis('off')
```

Take a look at St John's robes: they look pitch black in one image, gray in
another, and bright white in the third! This would result in a terrible MSE
score, even with perfect alignment.

Let's see what we can do with this. We start by splitting the plate into its
component channels:

```python
nrows = stained_glass.shape[0]
step = nrows // 3
channels = (stained_glass[:step],
            stained_glass[step:2*step],
            stained_glass[2*step:3*step])
channel_names = ['blue', 'green', 'red']
fig, axes = plt.subplots(1, 3)
for ax, image, name in zip(axes, channels, channel_names):
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(name)
```

First, we verify that the alignment indeed needs to be fine-tuned between the
three channels:

```python
blue, green, red = channels
original = np.dstack((red, green, blue))
fig, ax = plt.subplots()
ax.imshow(original)
ax.axis('off')
```

You can see by the color "halos" around objects in the image that the colors
are close to alignment, but not quite. Let's try to align them in the same
way that we aligned the astronaut image above, using the MSE. We use one color
channel, green, as the reference image, and align the blue and red channels to
that.

```python
print('*** Aligning blue to green ***')
tf = align(green, blue)
cblue = transform.warp(blue, tf, order=3)

print('** Aligning red to green ***')
tf = align(green, red)
cred = transform.warp(red, tf, order=3)

corrected = np.dstack((cred, green, cblue))
f, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(original)
ax0.set_title('Original')
ax1.imshow(corrected)
ax1.set_title('Corrected')
for ax in (ax0, ax1):
    ax.axis('off')
```

The alignment is a little bit better than with the raw images, because the red
and the green channels are correctly aligned, probably thanks to the giant
yellow patch of sky. However, the blue channel is still off, because the bright
spots of blue don't coincide with the green channel. That means that the MSE
will be lower when the channels are *mis*-aligned so that blue patches overlap
with some bright green spots.

We turn instead to a measure called *normalized mutual information* (NMI),
which measures correlations between the different brightness bands of the
different images. When the images are perfectly aligned, any object of uniform
color will create a large correlation between the shades of the different
component channels, and a correspondingly large NMI value. In a sense, NMI
measures how easy it would be to predict a pixel value of one image given the
value of the corresponding pixel in the other. It was defined in the paper:
Studholme, C., Hill, D.L.G., Hawkes, D.J.: An Overlap Invariant Entropy Measure
of 3D Medical Image Alignment. Patt. Rec. 32, 71–86 (1999):

$$
I(X, Y) = \frac{H(X) + H(Y)}{H(X, Y)},
$$

where $H(X)$ is the *entropy* of $X$, and $H(X, Y)$ is the joint
entropy of $X$ and $Y$. The numerator describes the entropy of the
two images, seen separately, and the denominator the total entropy if
they are observed together. Values can vary between 1 (maximally
aligned) and 2 (minimally aligned)[^mi_calc]. (See Chapter 5 for a
more in-depth discussion of entropy.)

[^mi_calc]: A quick handwavy explanation is that entropy is calculated
            from the histogram of the quantity under consideration.
            If $X = Y$, then the joint histogram $(X, Y)$ is diagonal,
            and that diagonal is the same as that of either $X$ or
            $Y$.  Thus, $H(X) = H(Y) = H(X, Y)$ and $I(X, Y) = 2$.

In Python code, this becomes:

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

Now we define a *cost function* to optimize, as we defined `cost_mse` above:

```python
def cost_nmi(param, reference_image, target_image):
    transformation = make_rigid_transform(param)
    transformed = transform.warp(target_image, transformation, order=3)
    return -normalized_mutual_information(reference_image, transformed)
```

Finally, we use this with our basinhopping-optimizing aligner:

```python
print('*** Aligning blue to green ***')
tf = align(green, blue, cost=cost_nmi)
cblue = transform.warp(blue, tf, order=3)

print('** Aligning red to green ***')
tf = align(green, red, cost=cost_nmi)
cred = transform.warp(red, tf, order=3)

corrected = np.dstack((cred, green, cblue))
f, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(original)
ax0.set_title('Original')
ax1.imshow(corrected)
ax1.set_title('Corrected')
for ax in (ax0, ax1):
    ax.axis('off')
```

What a glorious image! Realise that this artifact was created before color
photography existed! Notice God's pearly white robes, John's white beard,
and the white pages of the book held by Prochorus, his scribe — all of which
were missing from the MSE-based alignment.

Notice also the realistic gold of the candlesticks in the foreground.

## Registration in human brain imaging

This brings us to our final optimization in this chapter: aligning brain
images taken with extremely different techniques. MRI and PET imaging are
so different, that even normalized mutual information is insufficient for
some alignments.

```python
#TODO: attempt to align multimodal 3D images with NMI
```

In their 2000 paper, Pluim *et al.* showed that you can improve the properties
of the cost function by adding *gradient* information to the NMI metric. In
short, they measure the gradients magnitude and direction in both images (think
back to Chapter 3 and the Sobel filter), and score highly where the gradients
align, meaning they point in similar or opposite directions, but not at sharp
angles to each other.

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

```python
# TODO: align brain images with this alignment similarity function
```
