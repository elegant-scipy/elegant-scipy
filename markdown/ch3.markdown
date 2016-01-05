```python
# set up Py 2-3 compatibility
from __future__ import print_function, division
from six.moves import map, range, zip, filter
```

# Vighnesh Birodkar: build a region adjacency graph from nD images

You probably know that digital images are made up of *pixels*. These are
the light signal *sampled on a regular grid*. When computing
on images, we often deal with objects much larger than individual pixels.
In a landscape, the sky, earth, trees, rocks each span many
pixels. A common structure to represent these is the Region Adjacency Graph,
or RAG. It holds the properties of each region in the image, and the spatial
relationships between them. Building such a structure could be a complicated
affair, and even more difficult 
when images are not two-dimensional but 3D and even 4D, as is
common in microscopy, materials science, and climatology, among others. But
here we will show you how to produce an RAG in a few lines of code using NetworkX and
a generalized filter from SciPy's N-dimensional image processing submodule.

```python
import networkx as nx
import numpy as np
from scipy import ndimage as nd

def add_edge_filter(values, graph):
    current = values[0]
    neighbors = values[1:]
    for neighbor in neighbors:
        graph.add_edge(current, neighbor)
    return 0.

def build_rag(labels, image):
    g = nx.Graph()
    footprint = nd.generate_binary_structure(labels.ndim, connectivity=1)
    for j in range(labels.ndim):
        fp = np.swapaxes(footprint, j, 0)
        fp[0, ...] = 0
    _ = nd.generic_filter(labels, add_edge_filter, footprint=footprint,
                          mode='nearest', extra_arguments=(g,))
    for n in g:
        g.node[n]['total color'] = np.zeros(3, np.double)
        g.node[n]['pixel count'] = 0
    for index in np.ndindex(labels.shape):
        n = labels[index]
        g.node[n]['total color'] += image[index]
        g.node[n]['pixel count'] += 1
    return g
```

There are a few things going on here: images being represented as numpy arrays,
*filtering* of these images using `scipy.ndimage`, and building these into a
graph (network) using the NetworkX library. We'll go over these in turn.

# Images are numpy arrays

In the previous chapter, we saw that numpy arrays can efficiently represent
tabular data, and are a convenient way to perform computations on it.
It turns out that arrays are equally adept at representing images.

Here's how to create an image of white noise using just numpy, and display it
with matplotlib. First, we import the necessary packages, and use the `matplotlib
inline` IPython magic to make our images appear below the code:

```python
%matplotlib inline
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, cm
```

Next, we set the default matplotlib colormap and interpolation method:

```python
mpl.rcParams['image.cmap'] = 'gray'
mpl.rcParams['image.interpolation'] = None
```

Finally, "make some noise" and display it as an image:

```python
random_image = np.random.rand(500, 500)
plt.imshow(random_image);
```

This displays a numpy array as an image. The converse is also true: an image
can be considered "as" a numpy array. For this example we use the scikit-image
library, a collection of image processing tools built on top of NumPy and SciPy.

Here is PNG image from the scikit-image repository. It is a black and white
(sometimes called "grayscale") picture of some ancient Roman coins from
Pompeii, obtained from the Brooklyn Museum [^coins-source]:

![Coins](https://raw.githubusercontent.com/scikit-image/scikit-
image/v0.10.1/skimage/data/coins.png)

Here is the coin image loaded with scikit-image:

```python
from skimage import io
url_coins = 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.10.1/skimage/data/coins.png'
coins = io.imread(url_coins)
print("Type:", type(coins), "Shape:", coins.shape, "Data type:", coins.dtype)
plt.imshow(coins);
```

A grayscale image can be represented as a *2-dimensional array*, with each array
element containing the grayscale intensity at that position. So, **an image is
just a numpy array**.

Color images are a *3-dimensional* array, where the first two dimensions
represent the spatial positions of the image, while the final dimension represents
color channels, typically the three primary additive colors of red, green, and blue.
To show what we can do with these dimensions, let's play with this photo of an astronaut:

```python
url_astronaut = 'https://raw.githubusercontent.com/scikit-image/scikit-image/master/skimage/data/astronaut.png'
astro = io.imread(url_astronaut)
print("Type:", type(astro), "Shape:", astro.shape, "Data type:", astro.dtype)
plt.imshow(astro);
```

This image is *just numpy arrays*. Adding a green square to the image is easy
once you realize this, using simple numpy slicing:

```python
astro_sq = np.copy(astro)
astro_sq[50:100, 50:100] = [0, 255, 0]  # red, green, blue
plt.imshow(astro_sq);
```

You can also use a boolean mask:

```python
astro_sq = np.copy(astro)
sq_mask = np.zeros(astro.shape[:2], bool)
sq_mask[50:100, 50:100] = True
astro_sq[sq_mask] = [0, 255, 0]
plt.imshow(astro_sq);
```

**Exercise:** Create a function to draw a green grid onto a color image, and
apply it to the `astronaut` image of Eileen Collins (above). Your function should take
two parameters: the input image, and the grid spacing.
Use the following template to help you get started.

```python
def overlay_grid(image, spacing=128):
    """Return an image with a grid overlay, using the provided spacing.

    Parameters
    ----------
    image : array, shape (M, N, 3)
        The input image.
    spacing : int
        The spacing between the grid lines.

    Returns
    -------
    image_gridded : array, shape (M, N, 3)
        The original image with a grid superimposed.
    """
    image_gridded = image.copy()
    pass  # replace this line with your code...
    return image_gridded

# plt.imshow(overlay_grid(astro, 128));  # ... and uncomment this line to test your function
```

# Image filters

Filtering is one of the most fundamental and common operations in image
processing. You can filter an image to remove noise, to enhance features, or to
detect edges between objects in the image.

To understand filters, it's easiest to start with a 1D signal, instead of an image. For
example, you might measure the light arriving at your end of a fiber-optic cable.
If you *sample* the signal every millisecond (ms) for 100ms, you end up with an
array of length 100. Suppose that after 30ms the light signal is turned on, and
30ms later, it is switched off. You end up with a signal like this:

```python
sig = np.zeros(100, np.float) # 
sig[30:60] = 1  # signal = 1 during the period 30-60ms because light is observed
plt.plot(sig);
plt.ylim(-0.1, 1.1);
```

To find *when* the light is turned on, you can *delay* it by 1ms, then
*subtract* the delayed signal from the original, and finally *clip* this
difference to be nonzero.

```python
sigdelta = sig[:-1]  # sigd[0] equals sig[1], and so on
sigdiff = sig[1:] - sigdelta
sigon = np.clip(sigdiff, 0, np.inf)
print(1 + np.flatnonzero(sigon)[0], 'ms')
```

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

```python
diff = np.array([1, -1])
from scipy import ndimage as nd
dsig = nd.convolve(sig, diff)
plt.plot(dsig);
```

Signals are usually *noisy* though, not perfect as above:

```python
sig = sig + np.random.normal(0, 0.3, size=sig.shape)
plt.plot(sig);
```

The plain difference filter can amplify that noise:

```python
plt.plot(nd.convolve(sig, diff));
```

In such cases, you can add smoothing to the filter:

```python
smoothdiff = np.array([.5, 1.5, 3, 5, -5, -3, -1.5, -0.5]) / 10
```

This smoothed difference filter looks for an edge in the central position,
but also for that difference to continue. This is true in the case of a true
edge, but not in "spurious" edges caused by noise. We then assign a
weight of 0.5 to edges in the center, progressively decreasing as we move away
from it.
Check out the result:

```python
sdsig = nd.convolve(sig, smoothdiff)
plt.plot(sdsig);
```

(Note: this operation is called filtering because, in physical electrical
circuits, many of these operations are implemented by hardware that
lets certain kinds of current through, but not others; these components
are called filters. For example, a common filter that removes high-frequency
voltage fluctuations from a current is called a *low-pass filter*.)

Now that you've seen filtering in 1D, I hope you'll find it straightforward
to extend these concepts to 2D. Here's a 2D difference filter finding the
edges in the coins image:

```python
coins = coins.astype(float) / 255  # prevents overflow errors
diff2d = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]])
coins_edges = nd.convolve(coins, diff2d)
io.imshow(coins_edges);
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

```python
hsobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
vsobel = hsobel.T
coins_h = nd.convolve(coins, hsobel)
coins_v = nd.convolve(coins, vsobel)
coins_sobel = np.sqrt(coins_h**2 + coins_v**2)
plt.imshow(coins_sobel, cmap=plt.cm.YlGnBu);
plt.colorbar();
```

In addition to dot-products, implemented by `nd.convolve`, SciPy lets you
define a filter that is an *arbitrary function* of the points in a neighborhood,
implemented in `nd.generic_filter`. This can let you express arbitrarily complex
filters.

For example, suppose an image represents median house values in a county,
with a 100m x 100m resolution. The local council decides to tax house sales as
$10,000 plus 5% of the 90th percentile of house prices in a 1km radius. (So,
selling a house in an expensive neighborhood costs more.) With
`generic_filter`, we can produce the map of the tax rate everywhere in the map:

```python
from skimage import morphology
def tax(prices):
    return 10 + 0.05 * np.percentile(prices, 90)
house_price_map = (0.5 + np.random.rand(100, 100)) * 1e6
footprint = morphology.disk(radius=10)
tax_rate_map = nd.generic_filter(house_price_map, tax, footprint=footprint)
plt.imshow(tax_rate_map)
plt.colorbar()
```

**Exercise:** Conway's
[Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is a
seemingly simple construct in which "cells" on a regular square grid live or die
according to the cells in their immediate surroundings. At every timestep, we
determine the state of position (i, j) according to its previous state and that
of its 8 neighbors (above, below, left, right, and diagonals):

- a live cell with only one live neighbor or none dies.
- a live cell with two or three live neighbors lives on for another generation.
- a live cell with four or more live neighbors dies, as if from overpopulation.
- a dead cell with exactly three live neighbors becomes alive, as if by
  reproduction.

Although the rules sound like a contrived math problem, they in fact give rise
to incredible patterns, starting with gliders (small patterns of live cells
that slowly move in each generation) and glider guns (stationary patterns that
sprout off gliders), all the way up to prime number generator machines (see,
for example,
[this page](http://www.njohnston.ca/2009/08/generating-sequences-of-primes-in-conways-game-of-life/)),
and even
[simulating Game of Life itself](https://www.youtube.com/watch?v=xP5-iIeKXE8)!

Can you implement the Game of Life using `nd.generic_filter`?

# Graphs and the NetworkX library

To introduce you to graphs, we will reproduce some results from the paper 
["Structural properties of the *Caenorhabditis elegans* neuronal networks"](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001066), by Varshney *et al*, 2011.
Note that in this context the term "graph" is synonymous with "network", but not with "plot".
Mathematicians and computer scientists invented slightly different words to discuss these:
graph = network, vertex = node, and edge = link.
As most people do, we will be using these terms interchangeably.

You might be slightly more familiar with the network terminology: a network consists of
*nodes* and *links* between the nodes. Equivalently, a graph consists of *vertices* and
*edges* between the vertices. In NetworkX, you have `Graph` objects consisting of
`nodes` and `edges` between the nodes. Oh well.

Graphs are a natural representation for a bewildering variety of data. Pages on the world
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
import os
import xlrd  # Excel-reading library in Python

try:
    from urllib.request import urlopen  # getting files from the web, Py3
except ImportError:
    from urllib2 import urlopen  # getting files from the web, Py2

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
```

(Look at the WormAtlas page for a description of the different connection types.)

We use `weight` in a dictionary above because it is a special keyword for
edge properties in NetworkX. We then build the graph using NetworkX's
`DiGraph` class:

```python
import networkx as nx
wormbrain = nx.DiGraph()
wormbrain.add_edges_from(conn_edges)
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
implicated in how the worm responds to prodding: the AVA neurons link
the worm's front touch receptors (among others) to neurons responsible
for backward motion, while the PVC neurons link the rear touch receptors to
forward motion.

These neurons' high centrality feels like a bit of an artifact of their placement
controlling a large number of motor neurons. Yes, they are in many routes
from sensory neurons to motor neurons. But all of the motor neurons do essentially
the same thing, as hinted at by their generic names, VA 1-12. If we were to collapse
them into one, the high centrality of the "command" neurons AVA R and L, and
PVC R and L, might vanish. Returning to the rail lines example, suppose trains
between Grand Central Station in New York City and Washington DC's Union
Station could end up at one of 12 different platforms, *and we counted each of
those as a separate train line*. The betweenness centrality of Grand Central
would be inflated because from it you could get to Union Station platform 1,
platform 2, etc. That's not necessarily very interesting.

Varshney *et al* study the properties of a *strongly connected component*
of 237 neurons, out of a total of 279. In graphs, a
*connected component* is a set of nodes that are reachable by some path
through all the links. The connectome is a *directed* graph, meaning the
edges *point* from one node to the other, rather than merely connecting
them. In this case, a strongly connected component is one where all nodes
are reachable from each other by traversing links *in the correct direction*.
So A -> B -> C is not strongly connected, because there is no way to get to
A from B or C. but A -> B -> C -> A *is* strongly connected.

In a neuronal circuit, you can think of the strongly connected component
as the "brain" of the circuit, where the processing happens, while nodes
upstream of it are inputs, and nodes downstream are outputs.

> **Box**
> 
> The idea of cyclical neuronal circuits dates back to the 1950s. Here's a
> lovely paragraph about this idea from an article in *Nautilus*,
> "The Man Who Tried to Redeem the World With Logic", by Amanda Gefter:
> 
> If one were to see a lightning bolt flash on the sky, the eyes would send a signal to the brain, shuffling it through a chain of neurons. Starting with any given neuron in the chain, you could retrace the signal's steps and figure out just how long ago the lightning struck. Unless, that is, the chain is a loop. In that case, the information encoding the lightning bolt just spins in circles, endlessly. It bears no connection to the time at which the lightning actually occurred. It becomes, as McCulloch put it, "an idea wrenched out of time." In other words, a memory.

NetworkX makes straightforward work out of getting the largest strongly
connected component from our `wormbrain` network:

```python
sccs = nx.strongly_connected_component_subgraphs(wormbrain)
sccs = sorted(sccs, key=len, reverse=True)
giantscc = sccs[0]
print('The largest strongly connected component has %i nodes,' %
      giantscc.number_of_nodes(), 'out of %i total.' %
      wormbrain.number_of_nodes())
```

As noted in the paper, the size of this component is *smaller* than
expected by chance, demonstrating that the network is segregated into
input, central, and output layers.

Now we reproduce figure 6B from the paper, the survival function of the
in-degree distribution. First, compute the relevant quantities:

```python
in_degrees = list(wormbrain.in_degree().values())
in_deg_distrib = np.bincount(in_degrees)
avg_in_degree = np.mean(in_degrees)
cumfreq = np.cumsum(in_deg_distrib) / np.sum(in_deg_distrib)
survival = 1 - cumfreq
```

Then, plot using Matplotlib:

```python
plt.loglog(np.arange(1, len(survival) + 1), survival, c='b', lw=2)
plt.xlabel('in-degree distribution')
plt.ylabel('fraction of neurons with higher in-degree distribution')
plt.scatter(avg_in_degree, 0.0022, marker='v')
plt.text(avg_in_degree - 0.5, 0.003, 'mean=%.2f' % avg_in_degree, )
plt.ylim(0.002, 1.0)
plt.show()
```

**Exercise**: Use `scipy.optimize.curve_fit` to fit the tail of the
in-degree survival function to a power-law,
$f(d) \sim d^{-\gamma}, d > d_0$,
for $d_0 = 10$ (the red line in Figure 6B of the paper), and modify the plot
to include that line.

# Region adjacency graphs

I hope that the previous section gave you an idea of the power of graphs as a scientific
abstraction, and also how Python makes it easy to manipulate and analyse
them. Now we will study a special kind of graph, the region adjacency
graph, or RAG. This is a representation of an image that is useful for *segmentation*,
the division of images into meaningful regions (or *segments*). If you've seen
Terminator 2, you've seen segmentation:

![Terminator vision](https://raw.githubusercontent.com/scikit-image/skimage-tutorials/master/2014-scipy/images/terminator-vision.png)

Segmentation is one of those problems that humans do trivially, all the time,
without thinking, whereas computers have a really hard time of it. To
understand this difficulty, look at this image:

![Face (Eileen Collins)](http://i.imgur.com/ky5qwIS.png)

While you see a face, a computer only sees a bunch of numbers:

```
    58688888888888899998898988888666532121
    66888886888998999999899998888888865421
    66665566566689999999999998888888888653
    66668899998655688999899988888668665554
    66888899998888888889988888665666666543
    66888888886868868889998888666688888865
    66666443334556688889988866666666668866
    66884235221446588889988665644644444666
    86864486233664666889886655464321242345
    86666658333685588888866655659381366324
    88866686688666866888886658588422485434
    88888888888688688888866566686666565444
    88888888868666888888866556688666686555
    88888988888888888888886656888688886666
    88889999989998888888886666888888868886
    88889998888888888888886566888888888866
    88888998888888688888666566868868888888
    68888999888888888868886656888888888866
    68888999998888888688888655688888888866
    68888999886686668886888656566888888886
    88888888886668888888888656558888888886
    68888886665668888889888555555688888886
    86868868658668868688886555555588886866
    66688866468866855566655445555656888866
    66688654888886868666555554556666666865
    88688658688888888886666655556686688665
    68888886666888888988888866666656686665
    66888888845686888999888886666556866655
    66688888862456668866666654431268686655
    68688898886689696666655655313668688655
    68888898888668998998998885356888986655
    68688889888866899999999866666668986655
    68888888888866666888866666666688866655
    56888888888686889986868655566688886555
    36668888888868888868688666686688866655
    26686888888888888888888666688688865654
    28688888888888888888668666668686666555
    28666688888888888868668668688886665548
```

(Yes, your visual system is tuned enough to find faces that it sees the
face even in this blob of numbers! But I hope you get my point. Also, check
out the "Faces In Things" Tumblr.)

So the challenge is to make sense of those numbers, and where the
boundaries lie that divide the different parts of the image. A popular
approach is to find small regions (called superpixels) that
you're *sure* belong in the same segment, and then merge those according
to some more sophisticated rule.

As a simple example, suppose you want to segment out the tiger in this
picture, from the Berkeley Segmentation DataSet (BSDS) [^bsds-tiger]:

![BSDS-108073 tiger](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg)

A clustering algorithm, simple linear iterative clustering (SLIC) [^slic], can give
us a decent starting point. It is available in the scikit-image library.

```python
url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg'
tiger = io.imread(url)
from skimage import segmentation
seg = segmentation.slic(tiger, n_segments=30, compactness=40.0,
                        enforce_connectivity=True, sigma=3)
```

Scikit-image also has a function to *display* segmentations, which we use to
visualize the result of SLIC:

```python
from skimage import color
io.imshow(color.label2rgb(seg, tiger))
```

This shows that the tiger has been split in three parts, with the rest of the image
in the remaining segments.

A region adjacency graph (RAG) is a graph in which every node represents one
of the above regions, and an edge connects two nodes when they touch. For a
taste of what it looks like before we build one, we'll use the `draw_rag` function
from scikit-image — indeed, the library that contains this chapter's code snippet!

```python
from skimage.future import graph
g = graph.rag_mean_color(tiger, seg)
io.imshow(graph.draw_rag(seg, g, tiger, desaturate=True,
                         colormap=plt.cm.YlGnBu))
```

Here, you can see the nodes corresponding to each segment, and the edges
between adjacent segments. These are colored with the YlGnBu (yellow-green-blue)
colormap from matplotlib, according to the difference in color between the
two nodes.

The figure also shows the magic of thinking of segmentations as graphs: you can
see that edges between nodes within the tiger and those outside of it are darker
(higher-valued) than edges within the same object. Thus, if we can cut the
graph along those edges, we will get our segmentation! (Yes, I have chosen an easy
example for color-based segmentation, but the same principles hold true for
graphs with more complicated pairwise relationships!)

# Elegant ndimage

All the pieces are in place: you know about numpy arrays, image filtering,
generic filters, graphs, and region adjacency graphs. Let's build one to pluck
the tiger out of that picture!

The obvious approach is to use two nested for-loops to iterate over every pixel
of the image, look at the neighboring pixels, and checking for different labels:

```python
import networkx as nx
def build_rag(labels, image):
    g = nx.Graph()
    nrows, ncols = labels.shape
    for row in range(nrows):
        for col in range(ncols):
            current_label = labels[row, col]
            if not current_label in g:
                g.add_node(current_label)
                g.node[current_label]['total color'] = np.zeros(3, dtype=np.float)
                g.node[current_label]['pixel count'] = 0
            if row < nrows - 1 and labels[row + 1, col] != current_label:
                g.add_edge(current_label, labels[row + 1, col])
            if col < ncols - 1 and labels[row, col + 1] != current_label:
                g.add_edge(current_label, labels[row, col + 1])
            g.node[current_label]['total color'] += image[row, col]
            g.node[current_label]['pixel count'] += 1
    return g
```

This works, but if you want to segment a 3D image, you'll have to write a
different version:

```python
import networkx as nx
def build_rag_3d(labels, image):
    g = nx.Graph()
    nplns, nrows, ncols = labels.shape
    for pln in range(nplns):
        for row in range(nrows):
            for col in range(ncols):
                current_label = labels[pln, row, col]
                if not current_label in g:
                    g.add_node(current_label)
                    g.node[current_label]['total color'] = np.zeros(3, dtype=np.float)
                    g.node[current_label]['pixel count'] = 0
                if pln < nplns - 1 and labels[pln + 1, row, col] != current_label:
                    g.add_edge(current_label, labels[pln + 1, row, col])
                if row < nrows - 1 and labels[pln, row + 1, col] != current_label:
                    g.add_edge(current_label, labels[pln, row + 1, col])
                if col < ncols - 1 and labels[pln, row, col + 1] != current_label:
                    g.add_edge(current_label, labels[pln, row, col + 1])
                g.node[current_label]['total color'] += image[pln, row, col]
                g.node[current_label]['pixel count'] += 1
    return g
```

Both of these are pretty ugly and unwieldy, too. And difficult to extend:
if we want to count diagonally neighboring pixels as adjacent (that is,
[row, col] is "adjacent to" [row + 1, col + 1]), the code becomes even
messier. And if we want to analyze 3D video, we need yet another
dimension, and another level of nesting. It's a mess!

Enter Vighnesh's insight: SciPy's `generic_filter` function already does
this iteration for us! We used it above to compute an arbitrarily
complicated function on the neighborhood of every element of a numpy
array. Only now we don't want a filtered image out of the function: we
want a graph. It turns out that `generic_filter` lets you pass additional
arguments to the filter function, and we can use that to build the graph:

```python
import networkx as nx
import numpy as np
from scipy import ndimage as nd

def add_edge_filter(values, graph):
    current = values[0]
    neighbors = values[1:]
    for neighbor in neighbors:
        graph.add_edge(current, neighbor)
    return 0. # generic_filter requires a return value, which we ignore!

def build_rag(labels, image):
    g = nx.Graph()
    footprint = nd.generate_binary_structure(labels.ndim, connectivity=1)
    for j in range(labels.ndim):
        fp = np.swapaxes(footprint, j, 0)
        fp[0, ...] = 0  # zero out top of footprint on each axis
    _ = nd.generic_filter(labels, add_edge_filter, footprint=footprint,
                          mode='nearest', extra_arguments=(g,))
    for n in g:
        g.node[n]['total color'] = np.zeros(3, np.double)
        g.node[n]['pixel count'] = 0
    for index in np.ndindex(labels.shape):
        n = labels[index]
        g.node[n]['total color'] += image[index]
        g.node[n]['pixel count'] += 1
    return g
```

There's a few things to notice here:

- the loops are not nested several levels deep. This makes the code more
  compact, easier to take in in one go.
- the code works identically for 1D, 2D, 3D, or even 8D images!
- if we want to add support for diagonal connectivity, we just need to
  change the `connectivity` parameter to `nd.generate_binary_structure`
- `nd.generic_filter` iterates over array elements *with their neighbors*;
  use `numpy.ndindex` to simply iterate over array indices.

Overall, I think this is just a brilliant piece of code.

# Putting it all together: mean color segmentation

Now, we can use it to segment the tiger in the image above:

```python
g = build_rag(seg, tiger)
for n in g:
    node = g.node[n]
    node['mean'] = node['total color'] / node['pixel count']
for u, v in g.edges_iter():
    d = g.node[u]['mean'] - g.node[v]['mean']
    g[u][v]['weight'] = np.linalg.norm(d)
```

Each edge holds the difference between the average color of each segment.
We can now threshold the graph:

```python
def threshold_graph(g, t):
    to_remove = ((u, v) for (u, v, d) in g.edges(data=True)
                 if d['weight'] > t)
    g.remove_edges_from(to_remove)
threshold_graph(g, 80)
```

Finally, we use the numpy index-with-an-array trick we learned in chapter 1:

```
map_array = np.zeros(np.max(seg) + 1, int)
for i, segment in enumerate(nx.connected_components(g)):
    for initial in segment:
        map_array[initial] = i
segmented = map_array[seg]
plt.imshow(color.label2rgb(segmented, tiger));
```

Oops! Looks like the cat lost its tail!

Still, I think that's a nice demonstration of the capabilities of RAGs...
And the beauty with which SciPy and NetworkX make it feasible!

Many of these functions are available in the scikit-image library. If you
are interested in image analysis, check it out!

[^coins-source]: http://www.brooklynmuseum.org/opencollection/archives/image/15641/image
[^openworm]: http://www.openworm.org
[^file-url]: https://github.com/scikit-image/scikit-image/tree/master/skimage/io/util.py
[^nxdoc]: http://networkx.github.io/documentation/latest/reference/index.html
[^bwcdoc]: http://networkx.github.io/documentation/latest/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html
[^bsdstiger]: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images/color/108073.html
[^slic]: http://ivrg.epfl.ch/research/superpixels
