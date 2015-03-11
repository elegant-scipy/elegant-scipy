
# Images are numpy arrays

In the previous chapter, we saw that numpy arrays can efficiently represent
tabular data, as well as perform computations on it.

It turns out that arrays are equally adept at representing images.

Here's how to create an image of white noise using just numpy, and display it
with matplotlib. First, we import the necessary packages, and use the `matplotlib
inline` IPython magic:

    %matplotlib inline
    import numpy as np
    import matplotlib as mpl
    from matplotlib import pyplot as plt, cm

Next, a little bit of matplotlib parameter tuning so things display how we want them:

    mpl.rcParams['image.cmap'] = 'gray'
    mpl.rcParams['image.interpolation'] = None
    mpl.rcParams['figure.figsize'] = (16, 12)

Finally, "make some noise" and display it as an image:

    random_image = np.random.rand(500, 500)
    plt.imshow(random_image, cmap=cm.gray, interpolation='none');

This displays a numpy array "as" an image. The converse is also true: an image
can be considered "as" a numpy array. For this example we use the scikit-image
library, a collection of image processing tools built on top of NumPy and SciPy.

Here is PNG image from the scikit-image repository. It is a black and white
(sometimes called "grayscale") picture of some ancient Roman coins from
Pompeii, obtained from the Brooklyn Museum [^coins-source]:

![Coins](https://raw.githubusercontent.com/scikit-image/scikit-
image/v0.10.1/skimage/data/coins.png)

Here it is loaded with scikit-image:

    from skimage import io
    url_coins = 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.10.1/skimage/data/coins.png'
    coins = io.imread(url_coins)
    print("Type:", type(coins), "Shape:", coins.shape, "Data type:", coins.dtype)
    plt.imshow(coins)

A grayscale image can be represented as a *2-dimensional array*, with each array
element containing the grayscale intensity at that position. So, **an image is
just a numpy array**.

Color images are a *3-dimensional* array, where the first two dimensions
represent the spatial extent of the image, while the final dimension represents
color channels, typically the three primary colors of red, green, and blue:

    url_astronaut = 'https://raw.githubusercontent.com/scikit-image/scikit-image/master/skimage/data/astronaut.png'
    astro = io.imread(url_astronaut)
    print("Type:", type(astro), "Shape:", astro.shape, "Data type:", astro.dtype)
    plt.imshow(astro)

These images are *just numpy arrays*. Adding a green square to the image is easy
once you realize this, using simple numpy slicing:

    astro_sq = np.copy(astro)
    astro_sq[50:100, 50:100] = [0, 255, 0]  # red, green, blue
    plt.imshow(astro_sq)

You can also superimpose a grid on the image, using a boolean mask:

    astro_gr = np.copy(astro)
    astro_gr[128::128, :] = [0, 255, 0]
    astro_gr[:, 128::128] = [0, 255, 0]
    plt.imshow(astro_gr, interpolation='bicubic')

**Exercise:** Create a function to draw a red major/minor grid onto a color image, and
apply it to the `astronaut` image of Eileen Collins (above). Your function should take
three parameters: input image, major spacing, minor spacing. The major gridlines
should be 3 pixels thick, while the minor ones should be one pixel thick.

    def major_minor_grid(image, spacing_major=256, spacing_minor=128):
        """Return an image with a major/minor grid, using the provided spacings.
    
        Parameters
        ----------
        image : array, shape (M, N, 3)
            The input image
        spacing_major, spacing_minor : int
            The spacing of the major and minor gridlines.
    
        Returns
        -------
        image_gridded : array, shape (M, N, 3)
            The original image with a red grid superimposed.
        """"
        pass # fill in here

# Image filters

Filtering is one of the most fundamental and common image operations in image
processing. You can filter an image to remove noise, to enhance features, or to
detect edges between objects in the image.

To understand filters, it's easiest to start with a 1D signal, instead of an image. For
example, you might measure the light arriving at your end of a fiber-optic cable.
If you *sample* the signal every ten milliseconds for a second, you end up with an
array of length 100. Suppose that after 300ms the light signal is turned on, and
300ms later, it is switched off. You end up with a signal like this:

    sig = np.zeros(100, np.float)
    sig[30:60] = 1
    plt.plot(sig)
    plt.ylim(-0.1, 1.1)

To find *when* the light is turned on, you can *delay* it by, say, 10ms, then
*subtract* the delayed signal from the original, and finally *take the absolute
value* of this difference.

    sigdelta = sig[1:]  # sigd[0] equals sig[1], and so on
    sigdiff = sig[:-1] - sigdelta
    sigon = np.abs(sigdiff)
    print(np.nonzero(sigon) * 10, 'ms')

It turns out that that can all be accomplished by *convolving* the signal
with a *difference filter*. In convolution, at every point of the *signal*, we
place the *filter* and produce the dot-product of the (reversed) filter against
the signal values preceding that location:
$s'(t) = \sum_{j=t-\tau}^{t}{s(j)f(t-j)}$
where $s$ is the signal, $s'$ is the filtered signal, $f$ is the filter, and
$\tau$ is the length of the filter.

Now, think of what happens when the filter is (1, -1), the difference filter:
when adjacent values (u, v) of the signal are identical, the filter produces
-u + v = 0. But when v > u, the signal produces some positive value.

    diff = np.array([1, -1])
    from scipy import ndimage as nd
    dsig = nd.convolve(sig, diff)
    plt.plot(dsig)

Signals are usually *noisy* though, not perfect as above:

    sig = sig + np.random.normal(0, 0.05, size=sig.shape)
    plt.plot(sig)

The plain difference filter can amplify that noise:

    plt.plot(nd.convolve(sig, diff))

In such cases, you can add smoothing to the filter:

    smoothdiff = np.array([0.2, 0.8, -0.8, -0.2])

This smoothed difference filter looks for an edge in the central position,
but also for that difference to continue. This is true in the case of a true
edge, but not in "spurious" edges caused by noise. We then assign a
weight of 0.8 to edges in the center, and 0.2 to the edge extension.
Check out the result:

    sdsig = nd.convolve(sig, smoothdiff)
    plt.plot(sdsig)

Now that you've seen filtering in 1D, I hope you'll find it straightforward
to extend these concepts to 2D. Here's a 2D difference filter finding the
edges in the coins image:

** Generic filters: ** suppose you have an image that represents a map of
property values. Politicians come up with new tax scheme on house sales based
on the 90th percentile of house prices in a 1km radius. Why would you have
such a filter on hand? You can instead use a *generic filter*.

# Graphs and the NetworkX library

To introduce you to graphs, we will reproduce some results from the paper "Structural
properties of the *Caenorhabditis elegans* neuronal networks", by Varshney *et al*, 2011.
Note that in this context, "graphs" is synonymous not with "plots", but with "networks".
Mathematicians and computer scientists invented slightly different words to discuss these,
and, like most, we will be using them interchangeably.

You might be slightly more familiar with the network terminology: a network consists of
*nodes* and *links* between the nodes. Equivalently, a graph consists of *vertices* and
*edges* between the vertices. In NetworkX, you have `Graph` objects consisting of `nodes`
and `edges` between the nodes. Oh well.

Graphs are a natural representation for a bewildering array of data. Pages on the world
wide web, for example, can comprise nodes, while links between those pages can be,
well, links. Or, in so-called *transcriptional networks*, nodes represent genes and edges
connect genes that have a direct influence on each-other's expression.

In our example, we will represent neurons in the nematode worm's nervous system as
nodes, and place an edge between two nodes when a neuron makes a synapse with
another. (*Synapses* are the chemical connections through which neurons
communicate.) The worm is an awesome example of neural connectivity analysis
because every worm (of this species) has the same number of neurons (302), and the
connections between them are all known. This has resulted in the fantastic
[Openworm](http://www.openworm.org) project, which I encourage you to follow.

You can download the neuronal dataset in Excel format (yuck) from the WormAtlas
database [here](http://www.wormatlas.org/neuronalwiring.html#Connectivitydata).
The direct link to the data is:
[http://www.wormatlas.org/images/NeuronConnect.xls](http://www.wormatlas.org/images/NeuronConnect.xls)
Let's start by getting a list of rows out of the file:

    import xlrd  # Excel-reading library in Python
    import urllib2  # getting files from the web
    import tempfile as tmp

# Region adjacency graphs

# Elegant ndimage

To build a region adjacency graph for an image, you might use two nested for-
loops to iterate over every pixel of the image, looking at the neighboring
pixels, and checking for different labels:

(code)

This works, but if you want to segment a 3D image, you'll have to write a
different version:

(code)

Both of these are pretty ugly, too.

One way to simplify it is to test for 2D images, convert them to "flat" 3D
images, and use just one piece of 3D code to generate the graph:

This still feels a bit hacky and inelegant. What if we had a 3D video, and
wanted to do 4D?

(Vighnesh's code)

# Putting it all together: mean boundary segmentation


[^coins-source]: http://www.brooklynmuseum.org/opencollection/archives/image/15641/image
