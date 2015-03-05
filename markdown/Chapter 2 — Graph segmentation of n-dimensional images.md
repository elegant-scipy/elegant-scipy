
# Images are numpy arrays

In the previous chapter, we saw that numpy arrays can efficiently represent
tabular data, as well as perform computations on it.

It turns out that arrays are equally adept at representing images.

Here's how to create an image of white noise using just numpy, and display it
with
matplotlib:


    %matplotlib inline
    import numpy as np
    import matplotlib as mpl
    from matplotlib import pyplot as plt, cm
    
    mpl.rcParams['image.cmap'] = 'gray'
    mpl.rcParams['image.interpolation'] = 'nearest'
    mpl.rcParams['figure.figsize'] = (16, 12)
    
    random_image = np.random.rand(500, 500)
    plt.imshow(random_image, cmap=cm.gray, interpolation='nearest');


![png](Chapter%202%20%E2%80%94%20Graph%20segmentation%20of%20n-dimensional%20images_files/Chapter%202%20%E2%80%94%20Graph%20segmentation%20of%20n-dimensional%20images_2_0.png)


This shows us a numpy array "as" an image. The converse is also true: an image
can
be considered "as" a numpy array. For this example we use the scikit-image
library,
a collection of image processing tools built on top of NumPy and SciPy.

Here is PNG image from the scikit-image repository:

![Coins](https://raw.githubusercontent.com/scikit-image/scikit-
image/v0.10.1/skimage/data/coins.png)

Here it is loaded with scikit-image:


    from skimage import io
    url_coins = 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.10.1/skimage/data/coins.png'
    coins = io.imread(url_coins)
    print("Type:", type(coins), "Shape:", coins.shape, "Data type:", coins.dtype)
    plt.imshow(coins)

    Type: <class 'numpy.ndarray'> Shape: (303, 384) Data type: uint8





    <matplotlib.image.AxesImage at 0x107c64748>




![png](Chapter%202%20%E2%80%94%20Graph%20segmentation%20of%20n-dimensional%20images_files/Chapter%202%20%E2%80%94%20Graph%20segmentation%20of%20n-dimensional%20images_4_2.png)


So a grayscale image can be represented as a *2-dimensional array*, with each
array
element containing the grayscale intensity at that position.

Color images are a *3-dimensional* array, where the first two dimensions
represent the spatial extent of the image, while the final dimension represents
color channels, typically the three primary colors of red, green, and blue:


    url_astronaut = 'https://raw.githubusercontent.com/scikit-image/scikit-image/master/skimage/data/astronaut.png'
    astro = io.imread(url_astronaut)
    print("Type:", type(astro), "Shape:", astro.shape, "Data type:", astro.dtype)
    plt.imshow(astro)

    Type: <class 'numpy.ndarray'> Shape: (512, 512, 3) Data type: uint8





    <matplotlib.image.AxesImage at 0x108f65b70>




![png](Chapter%202%20%E2%80%94%20Graph%20segmentation%20of%20n-dimensional%20images_files/Chapter%202%20%E2%80%94%20Graph%20segmentation%20of%20n-dimensional%20images_6_2.png)


These images are *just numpy arrays*. Adding a green square to the image is easy
once you realize this, using simple numpy slicing:


    astro_sq = np.copy(astro)
    astro_sq[50:100, 50:100] = [0, 255, 0]  # red, green, blue
    plt.imshow(astro_sq)




    <matplotlib.image.AxesImage at 0x10ecef550>




![png](Chapter%202%20%E2%80%94%20Graph%20segmentation%20of%20n-dimensional%20images_files/Chapter%202%20%E2%80%94%20Graph%20segmentation%20of%20n-dimensional%20images_8_1.png)


You can also superimpose a grid on the image, using a boolean mask:


    astro_gr = np.copy(astro)
    astro_gr[128::128, :] = [0, 255, 0]
    astro_gr[:, 128::128] = [0, 255, 0]
    plt.imshow(astro_gr)




    <matplotlib.image.AxesImage at 0x10ed52208>




![png](Chapter%202%20%E2%80%94%20Graph%20segmentation%20of%20n-dimensional%20images_files/Chapter%202%20%E2%80%94%20Graph%20segmentation%20of%20n-dimensional%20images_10_1.png)


**Exercise:** Create a function to draw a major/minor grid onto an image, and
apply
it to the `astronaut` image of Eileen Collins (above). Your function should take
five parameters: input image, major spacing, minor spacing, major thickness, and
minor thickness.


# Image filters

(Introduction to image filters)

# Graphs and the NetworkX library

# Region adjacency graphs

# Elegant ndimage

To build a region adjacency graph for an image, you might use two nested for-
loops
to iterate over every pixel of the image, looking at the neighboring pixels, and
checking for different labels:

(code)

This works, but if you want to segment a 3D image, you'll have to write a
different
version:

(code)

Both of these are pretty ugly, too.

One way to simplify it is to test for 2D images, convert them to "flat" 3D
images,
and use just one piece of 3D code to generate the graph:



This still feels a bit hacky and inelegant. What if we had a 3D video, and
wanted to
do 4D?

(Vighnesh's code)

# Putting it all together: mean boundary segmentation


    
