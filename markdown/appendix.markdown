# Appendix A: Solutions to Exercises

## Chapter 2

**Exercise:** We leave you the exercise of implementing the approach described in the paper:

1. Take bootstrap samples (random choice with replacement) of the genes used to cluster the samples;
2. For each sample, produce a hierarchical clustering;
3. In a `(n_samples, n_samples)`-shaped matrix, store the number of times a sample pair appears together in a bootstrapped clustering.
4. Perform a hierarchical clustering on the resulting matrix.

This identifies groups of samples that frequently occur together in clusterings, regardless of the genes chosen.
Thus, these samples can be considered to robustly cluster together.

*Hint: use `np.random.choice` with `replacement=True` to create bootstrap samples of row indices.*

## Chapter 3

**Exercise:** Create a function to draw a green grid onto a color image, and
apply it to the `astronaut` image of Eileen Collins (above). Your function should take
two parameters: the input image, and the grid spacing.
Use the following template to help you get started.

```
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

**Exercise:** Conway's Game of Life.

Suggested by Nicolas Rougier.

Conway's [Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is a
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

**Solution:**

Code by: Nicolas Rougier (@rougier)

This is the game of life (cellular automata) in 10 lines of python with numpy.
The code is available from http://www.labri.fr/perso/nrougier/teaching/numpy.100/ (last question).
(code explanations is available from http://www.labri.fr/perso/nrougier/teaching/numpy/numpy.html)

**Exercise:** Use `scipy.optimize.curve_fit` to fit the tail of the
in-degree survival function to a power-law,
$f(d) \sim d^{-\gamma}, d > d_0$,
for $d_0 = 10$ (the red line in Figure 6B of the paper), and modify the plot
to include that line.

## Chapter 5

**Exercise:** Write an alternative way of computing the confusion matrix that only makes a single pass through `pred` and `gt`.

```
def confusion_matrix1(pred, gt):
    cont = np.zeros((2, 2))
    # your code goes here
    return cont
```

**Exercise:** Write a function to compute the confusion matrix in one pass, as
above, but instead of assuming two categories, infer the number of categories
from the input.

```
def general_confusion_matrix(pred, gt):
    n_classes = None  # replace `None` with something useful
    # your code goes here
    return cont
```

**Exercise**: write out the COO representation of the following matrix:

```
s2 = np.array([[0, 0, 6, 0, 0],
               [1, 2, 0, 4, 5],
               [0, 1, 0, 0, 0],
               [9, 0, 0, 0, 0],
               [0, 0, 0, 6, 7]])
```

**Exercise:** Compute the conditional entropy of month given rain. What is the
entropy of the month variable? (Ignore the different number of days in a
month.) Which one is greater? (*Hint:* the probabilities in the table are
the conditional probabilities of rain given month.)

```
prains = [25, 27, 24, 18, 14, 11, 7, 8, 10, 15, 18, 23]
prains = [p / 100 for p in prains]
pshine = [1 - p for p in prains]
p_rain_g_month = np.array((prains, pshine)).T
# replace 'None' below with expression for non-conditional contingency
# table. Hint: the values in the table must sum to 1.
p_rain_month = None
# Add your code below to compute H(M|R) and H(M)
pass
```

**Exercise:** Segmentation in practice

Try finding the best threshold for a selection of other images from the [Berkeley Segmentation Dataset and Benchmark](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).
Using the mean or median of those thresholds, then go and segment a new image. Did you get a reasonable segmentation?
