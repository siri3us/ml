# -*- coding: utf-8 -*-

import numpy as np

def affine_forward(x, w, b, debug=False, layer_name=''):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases,  of shape (M,)
    - debug: If True checks input types, nans and infs

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    if debug:
        assert isinstance(x, np.ndarray)
        assert isinstance(w, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert x.ndim >= 2
    if x.ndim > 2:
        x = x.reshape((x.shape[0], -1))
    out = np.dot(x, w) + b[None, :]
    if debug:
        assert not np.any(np.isnan(out)), 'Output of affine layer {} contains nans.'.format(layer_name)
        assert not np.any(np.isinf(out)), 'Output of affine layer {} contains infs.'.format(layer_name)
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx = np.dot(dout, w.T)     # [N, M] x [D, M].T = [N, D]
    dw = np.dot(x.T, dout)     # [D, N] x [N, M] = [D, M]
    db = np.sum(dout, axis=0)  # [N, M] -> [M,]
    return dx, dw, db
    
    
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = x.copy()
    out[x < 0] = 0
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    dx = dout.copy()
    dx[x < 0] = 0
    return dx


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y, debug=False):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)   
    log_probs = np.maximum(shifted_logits - np.log(Z), -100)
    probs = np.maximum(np.exp(log_probs), 1e-40)
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    if debug: 
        assert not np.isnan(loss)
        assert not np.isinf(loss)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
    
    
def batchnorm_forward(X, gamma, beta, bn_param, debug=False):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var  = momentum * running_var  + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps  = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = X.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=X.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=X.dtype))

    if debug:
        assert isinstance(gamma, np.ndarray)
        assert isinstance(beta, np.ndarray)
    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(X, axis=0)
        sample_var = np.var(X, axis=0)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        X_norm = (X - sample_mean[None, :]) / np.sqrt(sample_var + eps)[None, :]
        out = gamma[None, :] * X_norm + beta[None, :]
        cache = X, X_norm, sample_mean, sample_var, gamma, beta, eps
        if debug:
            assert isinstance(X_norm, np.ndarray)
            assert isinstance(sample_var, np.ndarray)
    elif mode == 'test':
        out = gamma[None, :] * ((X - running_mean[None, :]) / (np.sqrt(running_var[None, :]+ eps) )) + beta[None, :]
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    if debug:
        assert isinstance(out, np.ndarray)

    return out, cache


def batchnorm_backward(dout, cache, debug=False):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    X, X_norm, mean, var, gamma, beta, eps = cache
    B, D = dout.shape
    assert dout.ndim == 2
    assert dout.shape == X_norm.shape
    
    std_inv = 1.0 / (np.sqrt(var + eps))
    X_mu = X - mean[None, :]
    
    dLY = dout
    assert dLY.shape == (B, D)

    dLZ = np.multiply(gamma[None, :], dLY) # B x D
    assert dLZ.shape == (B, D)

    dLsigma2 = -0.5 * std_inv**3 * np.sum(np.multiply(X_mu, dLZ), axis=0)
    assert dLsigma2.shape == (D, )

    dLmu = -np.sum(dLZ * std_inv[None, :], axis=0) - (dLsigma2 * np.mean(2 * X_mu, axis=0))
    assert dLmu.shape == (D, )

    dgamma = np.sum(np.multiply(X_norm, dLY), axis=0)
    assert dgamma.shape == (D, )

    dbeta = np.sum(dLY, 0)
    assert dbeta.shape == (D, )

    dx = dLZ * std_inv[None, :] + 2.0 * dLsigma2[None, :] * X_mu / B + (1.0 / B) * dLmu[None, :]

    if debug:
        assert isinstance(dgamma, np.ndarray)
        assert isinstance(dbeta, np.ndarray)
        assert isinstance(dx, np.ndarray)
        assert dgamma.shape == (D,)
        assert dbeta.shape == (D,)
        assert dx.shape == (B, D)
    return dx, dgamma, dbeta

def batchnorm_backward_alt(output_grad, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    X, X_norm, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = X.shape
    X_grad = gamma / np.sqrt(sample_var + eps)[None, :] * ((output_grad - np.mean(output_grad, axis=0)[None, :]) - X_norm * np.mean(np.multiply(X_norm, output_grad), axis=0)[None, :]) 
    beta_grad = np.sum(output_grad, axis=0)
    gamma_grad = np.sum(np.multiply(X_norm, output_grad), axis=0)
    return X_grad, gamma_grad, beta_grad


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p = dropout_param['p']
    mode = dropout_param['mode']
    gen = dropout_param['gen']
    seed = dropout_param.get('seed', None)
    if seed is not None:
        gen = np.random.RandomState(seed)
    mask = None
    out = None

    if mode == 'train':
        mask = gen.choice([0., 1.], size=x.shape, p=[p, 1 - p])
        out = np.multiply(x, mask)
    elif mode == 'test':
        out = (1 - p) * x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    p = dropout_param['mode']
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = np.multiply(mask, dout)
    elif mode == 'test':
        dx = (1 - p) * dout
    return dx
