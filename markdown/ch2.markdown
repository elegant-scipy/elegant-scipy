
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
        """
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

(Note: this operation is called filtering because, in physical electrical
circuits, many of these operations are implemented by hardware that
lets certain kinds of currents through, but not others; these components
are called filters.)

Now that you've seen filtering in 1D, I hope you'll find it straightforward
to extend these concepts to 2D. Here's a 2D difference filter finding the
edges in the coins image:

```python
diff2d = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]])
coins_edges = nd.convolve(coins, diff2d)
io.imshow(coins_edges)
```

The principle is the same as the 1D filter: at every point in the image, place the
filter, compute the dot-product of the filter's values with the image values, and
place the result at the same location in the output image. And, as with the 1D
difference filter, when the filter is placed on a location with little variation, the
dot-product cancels out to zero, whereas, placed on a location where the
image brightness is changing, the values multiplied by 1 will be different from
those multiplied by -1, and the filter's output will be a positive or negative
value (depending on whether the image is brighter towards the bottom-right
or top-left at that point).

Just as with the 1D filter, you can get more sophisticated and smooth out
noise right within the filter. The *Sobel* filter is designed to do just that:

    hsobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    vsobel = hsobel.T
    coins_h = nd.convolve(coins, hsobel)
    coins_v = nd.convolve(coins, vsobel)
    coins_sobel = np.sqrt(coins_h**2 + coins_v**2)
    io.imshow(coins_sobel)

In addition to dot-products, implemented by `nd.convolve`, SciPy lets you
define a filter that is an *arbitrary function* of the points in a neighborhood,
implemented in `nd.generic_filter`. This can let you express arbitrarily complex
filters.

For example, suppose an image represents median house values in a county,
with a 100m x 100m resolution. The local council decides to tax house sales as
$10,000 plus 5% of the 90th percentile of house prices in a 1km radius. (So,
selling a house in an expensive neighborhood costs more.) With
`generic_filter`, we can produce the map of the tax rate everywhere in the map:

    from skimage import morphology
    def tax(prices):
        return 10 + 0.05 * np.percentile(prices, 90)
    footprint = morphology.disk(radius=10)
    tax_rate_map = nd.generic_filter(house_price_map, footprint, tax)

# Graphs and the NetworkX library

To introduce you to graphs, we will reproduce some results from the paper "Structural
properties of the *Caenorhabditis elegans* neuronal networks", by Varshney *et al*, 2011.
Note that in this context, "graphs" is synonymous not with "plots", but with "networks".
Mathematicians and computer scientists invented slightly different words to discuss these,
and, as most do, we will be using them interchangeably.

You might be slightly more familiar with the network terminology: a network consists of
*nodes* and *links* between the nodes. Equivalently, a graph consists of *vertices* and
*edges* between the vertices. In NetworkX, you have `Graph` objects consisting of
`nodes` and `edges` between the nodes. Oh well.

Graphs are a natural representation for a bewildering array of data. Pages on the world
wide web, for example, can comprise nodes, while links between those pages can be,
well, links. Or, in so-called *transcriptional networks*, nodes represent genes and edges
connect genes that have a direct influence on each other's expression.

In our example, we will represent neurons in the nematode worm's nervous system as
nodes, and place an edge between two nodes when a neuron makes a synapse with
another. (*Synapses* are the chemical connections through which neurons
communicate.) The worm is an awesome example of neural connectivity analysis
because every worm (of this species) has the same number of neurons (302), and the
connections between them are all known. This has resulted in the fantastic Openworm
project [^openworm], which I encourage you to follow.

You can download the neuronal dataset in Excel format (yuck) from the WormAtlas
database at [http://www.wormatlas.org/neuronalwiring.html#Connectivitydata](http://www.wormatlas.org/neuronalwiring.html#Connectivitydata).
The direct link to the data is:
[http://www.wormatlas.org/images/NeuronConnect.xls](http://www.wormatlas.org/images/NeuronConnect.xls)
Let's start by getting a list of rows out of the file. An elegant pattern from Tony
Yu [^file-url] enables us to open a remote URL as a local file:

```python
import xlrd  # Excel-reading library in Python
from urllib.request import urlopen  # getting files from the web
import tempfile
from contextlib import contextmanager
@contextmanager
def url2filename(url):
    _, ext = os.path.splitext(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        remote = urlopen(url)
        f.write(remote.read())
    try:
        yield f.name
    finally:
        os.remove(f.name)
connectome_url = "http://www.wormatlas.org/images/NeuronConnect.xls"
with url2filename(connectome_url) as fin:
    sheet = xlrd.open_workbook(fin).sheet_by_index(0)
    conn = [sheet.row_values(i) for i in range(1, sheet.nrows)]
```

`conn` now contains a list of connections of the form:

[Neuron1, Neuron2, connection type, strength]

We are only going to examine the connectome of chemical synapses, so we filter
out other synapse types as follows:

```python
conn_edges = [(n1, n2, {'weight': s}) for n1, n2, t, s in conn
              if t.startswith('S')]
```python

Finally, we build the graph using NetworkX's `DiGraph` class:

```python
import networkx as nx
wormbrain = nx.DiGraph()
wormbrain.add_edges_from(conn)
```

We can now examine some of the properties of this network. One of the
first things researchers ask about directed networks is which nodes are
the most critical to information flow within it. Nodes with high
*betweenness centrality* are those that belong the shortest path between
many different pairs of nodes. Think of a rail network: certain stations will
connect to many lines, so that you will be forced to change lines there
for many different trips. They are the ones with high betweenness
centrality.

With networkx, we can find similarly important neurons with ease. In the
networkx API documentation [^nxdoc], under "centrality", the docstring
for `betweenness_centrality` [^bwcdoc] specifies a function that takes a
graph as input and returns a dictionary mapping node IDs to betweenness
centrality values (floating point values).

```python
centrality = nx.betweenness_centrality(wormbrain)
```

Now we can find the neurons with highest centrality using the Python built-in
function `sorted`:

```python
central = sorted(centrality, key=centrality.__getitem__, reverse=True)
print(central[:5])
```

This returns the neurons AVAR, AVAL, PVCR, PVT, and PVCL, which have been
implicated in how the worm responds to external stimuli: the AVA neurons link
the worm's front touch receptors (among others) to motor neurons responsible
for backward motion, while the PVC neurons link the rear touch receptors to
forward motion.

These neurons' high centrality is a bit of an artifact of their placement
controlling a large number of motor neurons. Yes, they are in many routes
from sensory neurons to motor neurons. But all of the motor neurons do the same
thing, as indicated by their generic names, VA 1-12. If we were to collapse
them into one, the high centrality of the "command" neurons AVA R and L, and
PVC R and L, would vanish. How do we study this systematically?

Varshney *et al* study the properties of a *strongly connected component*
of 237 neurons, out of a total of 279. In graphs, a
*connected component* is a set of nodes that are reachable by some path
through all the links. the connectome is a *directed* graph, meaning the
edges *point* from one node to the other, rather than merely connecting
them. in this case, a strongly connected component is one where all nodes
are reachable from each other by traversing links *in the correct direction*.
so a -> b -> c is not strongly connected, because there is no way to get to
a from b or c. but a -> b -> c -> a *is* strongly connected.

In a neuronal circuit, you can think of the strongly connected component
as the "brain" of the circuit, where the processing happens, while nodes
upstream of it are inputs, and nodes downstream are outputs.

> **box**
> the idea of cyclical neuronal circuits dates back to the 1950s. here's a
> lovely paragraph about this idea from the article
> "the man who tried to redeem the world with logic", by **author**:
> > if one were to see a lightning bolt flash on the sky, the eyes would send a signal to the brain, shuffling it through a chain of neurons. starting with any given neuron in the chain, you could retrace the signal's steps and figure out just how long ago the lightning struck. unless, that is, the chain is a loop. in that case, the information encoding the lightning bolt just spins in circles, endlessly. it bears no connection to the time at which the lightning actually occurred. it becomes, as mcculloch put it, "an idea wrenched out of time." in other words, a memory.



# region adjacency graphs

i hope that chapter gave you an idea of the power of graphs as a scientific
abstraciton. now we will study a special kind of graph, the region adjacency
graph, or rag, a representation of an image useful for *segmentation*, the
division of images into meaningful regions (or *segments*). if you've seen
terminator 2, you've seen segmentation:

**terminator vision image**




# elegant ndimage

to build a region adjacency graph for an image, you might use two nested for-
loops to iterate over every pixel of the image, looking at the neighboring
pixels, and checking for different labels:

(code)

this works, but if you want to segment a 3d image, you'll have to write a
different version:

(code)

both of these are pretty ugly, too.

one way to simplify it is to test for 2d images, convert them to "flat" 3d
images, and use just one piece of 3d code to generate the graph:

this still feels a bit hacky and inelegant. what if we had a 3d video, and
wanted to do 4d?

(vighnesh's code)

# putting it all together: mean boundary segmentation


[^coins-source]: http://www.brooklynmuseum.org/opencollection/archives/image/15641/image
[^openworm]: http://www.openworm.org
[^file-url]: https://github.com/scikit-image/scikit-image/tree/master/skimage/io/util.py
[^nxdoc]: http://networkx.github.io/documentation/latest/reference/index.html
[^bwcdoc]: http://networkx.github.io/documentation/latest/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html
