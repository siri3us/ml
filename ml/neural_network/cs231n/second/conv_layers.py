#-*- coding: utf-8 -*-

from .layers import *
from .conv_layers_fast import *

from itertools import product
import numpy as np

def conv_forward_naive(X, W, biases, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - X: Input data of shape (N, C, H, W)
    - W: Filter weights of shape (F, C, HH, WW)
    - biases: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (X, W, biases, conv_param)
    """
    
    # Obtaining 'pad'
    pad = conv_param['pad']
    pad_h = (pad, pad)
    pad_w = (pad, pad)
    # Obtaining 'stride'
    stride = conv_param['stride']
    
    # No shape checks used here for now
    n_samples, n_channels, image_h, image_w = X.shape
    n_filters, n_channels, filter_h, filter_w = W.shape
    n_filters, = biases.shape
    
    # Sizes
    padding = ((0, 0), pad_h, pad_w)
    image_size  = (n_channels, image_h, image_w)
    filter_size = (n_channels, filter_h, filter_w)
    upp_pad, low_pad = pad_h;    padded_image_h = image_h + upp_pad + low_pad
    left_pad, right_pad = pad_w; padded_image_w = image_w + left_pad + right_pad
    padded_image_size = (n_channels, padded_image_h, padded_image_w)
    
    # Finding convolutions
    assert (padded_image_h - filter_h) % stride == 0, 'Incorrect height padding'
    assert (padded_image_w - filter_w) % stride == 0, 'Incorrect width padding'
    output_h = 1 + (padded_image_h - filter_h) // stride
    output_w = 1 + (padded_image_w - filter_w) // stride
    out = np.zeros((n_samples, n_filters, output_h, output_w), dtype=np.float64)
    
    W_flattened = W.reshape((n_filters, -1))
    for n_sample in range(n_samples):
        image = np.pad(X[n_sample], padding, mode='constant')
        assert image.shape == padded_image_size
        for new_i, i in enumerate(range(0, padded_image_h - filter_h + 1, stride)):
            for new_j, j in enumerate(range(0, padded_image_w - filter_w + 1, stride)):
                col = image[:, i:i + filter_h, j:j + filter_w].flatten()
                assert col.shape == (n_channels * filter_h * filter_w,)
                out[n_sample, :, new_i, new_j] = np.sum(np.multiply(col[None, :], W_flattened), axis=1) + biases
   
    cache = (X, W, biases, conv_param)
    return out, cache
    
    
def conv_backward_naive(Y_grad, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - output_grad Upstream derivatives.
    - cache: A tuple of (X, W, b, conv_param) as in conv_forward_naive:
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
          - 'pad': The number of pixels that will be used to zero-pad the input.
     
    Returns a tuple of:
    - X_grad: Gradient with respect to X
    - W_grad: Gradient with respect to W
    - b_grad: Gradient with respect to biases
    """
    X = cache[0]
    W = cache[1]
    b = cache[2]
    conv_param = cache[3]
    
    # Obtaining 'pad'
    pad = conv_param['pad']
    # Obtaining 'stride'
    stride = conv_param['stride']
    
    n_samples, n_filters, output_h, output_w = Y_grad.shape
    n_samples, n_channels, image_h,  image_w = X.shape
    n_filters, = b.shape
    n_filters, n_channels, filter_h, filter_w = W.shape

    X_grad = np.zeros(X.shape, dtype=np.float64)
    W_grad = np.zeros(W.shape, dtype=np.float64)
    b_grad = np.zeros(b.shape, dtype=np.float64)
    
    # Finding X_grad
    for n_sample in range(n_samples):
        for n_channel in range(n_channels):
            for h, w in product(range(image_h), range(image_w)):
                # Finding derivative over X[n_sample, n_channel, h, w]
                h += pad; w += pad # Coordinates in the padded X
                i_low = int(np.ceil((h - filter_h + 1) / stride)); i_max = h // stride
                j_low = int(np.ceil((w - filter_w + 1) / stride)); j_max = w // stride
                i_range = range(max(0, i_low), min(output_h - 1, i_max) + 1)
                j_range = range(max(0, j_low), min(output_w - 1, j_max) + 1)
                for i, j in product(i_range, j_range):
                    a = h - i * stride
                    b = w - j * stride
                    assert (a >= 0) & (a < filter_h)
                    assert (b >= 0) & (b < filter_w)
                    assert (i >= 0) & (i < output_h)
                    assert (j >= 0) & (j < output_w)
                    X_grad[n_sample, n_channel, h - pad, w - pad] +=\
                        np.sum(W[np.arange(n_filters), n_channel, a, b] * Y_grad[n_sample, np.arange(n_filters), i, j])
    
    X_padded = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    assert X_padded.shape == (n_samples, n_channels, 2 * pad + image_h, 2 * pad + image_w)
    
    for n_filter in range(n_filters): 
        for n_channel in range(n_channels):
            for h, w in product(range(filter_h), range(filter_w)):
                # Finding derivative over W[n_filter, n_channel, h, w]
                for i, j in product(range(output_h), range(output_w)):
                    W_grad[n_filter, n_channel, h, w] +=\
                        np.sum(X_padded[:, n_channel, i * stride + h, j * stride + w] * Y_grad[:, n_filter, i, j])
                    # Finding derivative over b[n_filter]
        b_grad[n_filter] += np.sum(Y_grad[:, n_filter, :, :])
    
    return X_grad, W_grad, b_grad    


def max_pool_forward_naive(X, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - X: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - Y: Output data
    - cache: (X, I, pool_param)
    """
    
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    n_samples, n_channels, image_h, image_w = X.shape
    output_h = float(image_h - pool_height) / stride + 1
    output_w = float(image_w - pool_width) / stride + 1
    assert output_h == int(output_h)
    assert output_w == int(output_w)
    output_h = int(output_h)
    output_w = int(output_w)
    Y = np.zeros((n_samples, n_channels, output_h, output_w), dtype=np.float64)
    I = np.zeros((n_samples, n_channels, output_h, output_w, 2), dtype=np.int32)
    for n_sample, n_channel in product(range(n_samples), range(n_channels)):
        X_slice = X[n_sample, n_channel]
        Y_slice = Y[n_sample, n_channel]
        I_slice = I[n_sample, n_channel]
        for i, j in product(range(output_h), range(output_w)):
            x = X_slice[i * stride : i * stride + pool_height, j * stride : j * stride + pool_width]
            assert x.shape == (pool_height, pool_width)
            h, w = np.unravel_index(np.argmax(x), x.shape)
            Y_slice[i, j] = x[h, w]
            assert x[h, w] == np.max(x)
            I_slice[i, j] = np.array([h + i * stride, w + j * stride])
    cache = (X, I, pool_param)
    return Y, cache


def max_pool_backward_naive(Y_grad, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - X_grad: Gradient with respect to x
    """
    X = cache[0]
    I = cache[1]
    pool_param = cache[2]
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    n_samples, n_filters, output_h, output_w = Y_grad.shape
    n_samples, n_channels, image_h, image_w = X.shape
    
    X_grad = np.zeros(X.shape, dtype=np.float64)
    for n_sample, n_channel in product(range(n_samples), range(n_channels)):
        X_grad_slice = X_grad[n_sample, n_channel]
        Y_grad_slice = Y_grad[n_sample, n_channel]
        I_slice = I[n_sample, n_channel]
        for i, j in product(range(output_h), range(output_w)):
            h, w = I_slice[i, j]
            X_grad_slice[h, w] = Y_grad_slice[i, j]
            
    return X_grad


def spatial_batchnorm_forward(X, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - Y: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    n_samples, n_channels, image_h, image_w = X.shape
    assert gamma.shape == beta.shape
    assert gamma.shape == (n_channels,)
    
    X_2d = np.reshape(np.transpose(X, [0, 2, 3, 1]), (-1, n_channels))
    assert X_2d.shape == (n_samples * image_h * image_w, n_channels)
    Y_2d, cache = batchnorm_forward(X_2d, gamma, beta, bn_param)
    Y = np.transpose(np.reshape(Y_2d, (n_samples, image_h, image_w, n_channels)), [0, 3, 1, 2])
    return Y, cache


def spatial_batchnorm_backward(Y_grad, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - Y_grad: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - X_grad: Gradient with respect to inputs, of shape (N, C, H, W)
    - gamma_grad: Gradient with respect to scale parameter, of shape (C,)
    - beta_grad: Gradient with respect to shift parameter, of shape (C,)
    """
    n_samples, n_channels, image_h, image_w = Y_grad.shape
    Y_grad_2d = np.reshape(np.transpose(Y_grad, [0, 2, 3, 1]), (-1, n_channels))
    X_grad_2d, gamma_grad, beta_grad = batchnorm_backward(Y_grad_2d, cache)
    X_grad = np.transpose(np.reshape(X_grad_2d, (n_samples, image_h, image_w, n_channels)), [0, 3, 1, 2])
    return X_grad, gamma_grad, beta_grad


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
    

