# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib

def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs: N x H X W X C array of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
    take as input an H x W x D array and return a (one-dimensional) array of
    length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 0:
            print('Done extracting features for %d / %d images' % (i, num_images))
    return imgs_features

def rgb2gray(rgb):
    """Convert RGB image to grayscale

    Parameters:
      rgb : RGB image

    Returns:
      gray : grayscale image
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def hog_feature(im, n_orientations=9, scell=(8, 8), sblock=(2, 2), full_circle=False):
    """Compute Histogram of Gradient (HOG) feature for an image

       Modified from skimage.feature.hog
       http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

     Reference:
       Histograms of Oriented Gradients for Human Detection
       Navneet Dalal and Bill Triggs, CVPR 2005

    Parameters:
      im : an input grayscale or rgb image

    Returns:
      feat: Histogram of Gradient (HOG) feature
    """
  
    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.atleast_2d(im)

    H, W   = image.shape # image size
    cx, cy = scell #number of gradient bins
    bx, by = sblock #pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    
    gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
    
    grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
    grad_ori = np.maximum(np.arctan2(gy, gx) + np.pi, 0) # [0, 2pi)
    angle_range = (0, np.pi)
    assert np.all(grad_ori < 2 * np.pi + 1e-10)
    grad_ori = np.minimum(grad_ori, 2 * np.pi - 1e-10)
    if not full_circle:
        grad_ori[grad_ori >= np.pi] -= np.pi # [0, pi)
        angle_range = (0, np.pi)
    angle_min, angle_max = angle_range
    
    n_cellsx = int(np.floor(W / cx))  # number of cells in x
    n_cellsy = int(np.floor(H / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, n_orientations))
    for i in range(n_orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori <  angle_max / n_orientations * (i + 1),
                                grad_ori, -1)
        temp_ori = np.where(grad_ori >= angle_max / n_orientations * i,
                                temp_ori, -1)
        # select magnitudes for those orientations
        cond2 = temp_ori > -1
        temp_mag = np.where(cond2, grad_mag, 0)
        filtered = uniform_filter(temp_mag, size=(cx, cy))[cx-1::cx, cy-1::cy]
        orientation_histogram[:,:,i] = filtered
    return orientation_histogram.ravel()

class HogCalculator:
    def __init__(self, n_orientations, scell=(8, 8), sblock=(2, 2), full_circle=False):
        self.n_orientations = n_orientations
        self.scell = scell
        self.sblock = sblock
        self.full_circle = full_circle
    def __call__(self, im):
        return hog_feature(im, self.n_orientations, self.scell, self.sblock, self.full_circle)

def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
    1D vector of length nbin giving the color histogram over the hue of the
    input image.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # return histogram
    return imhist
