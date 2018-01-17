# -*- coding: utf-8 -*-

import numpy as np

def sgd(w, dw, config={}):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    config.setdefault('learning_rate', 1e-2)
    next_w = w - config['learning_rate'] * dw
    return next_w, config


def sgd_momentum(w, dw, config={}):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    lr = config.setdefault('learning_rate', 1e-2)
    momentum = config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w).astype(config.setdefault('dtype', np.float64), copy=False))
    v *= momentum
    v -= lr * dw
    next_w = w + v
    config['velocity'] = v
    return next_w, config


def rmsprop(x, dx, config={}):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    learning_rate = config.setdefault('learning_rate', 1e-3)
    decay_rate = config.setdefault('decay_rate', 0.99)
    epsilon = config.setdefault('epsilon', 1e-8)
    cache = config.setdefault('cache', np.zeros_like(x).astype(config.setdefault('dtype', np.float64), copy=False))
    
    cache *= decay_rate
    cache += (1 - decay_rate) * (dx ** 2)
    next_x = x - learning_rate * dx / (np.sqrt(cache) + epsilon)
    config['cache'] = cache
    return next_x, config


def adam(x, dx, config={}):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    learning_rate = config.setdefault('learning_rate', 1e-3)
    beta1 = config.setdefault('beta1', 0.9)
    beta2 = config.setdefault('beta2', 0.999)
    epsilon = config.setdefault('epsilon', 1e-8)
    m = config.setdefault('m', np.zeros_like(x).astype(config.setdefault('dtype', np.float64), copy=False))
    v = config.setdefault('v', np.zeros_like(x).astype(config.setdefault('dtype', np.float64), copy=False))
    t = config.setdefault('t', 0)
    
    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * (dx ** 2)
    mt = m / (1 - beta1 ** (t + 1))
    vt = v / (1 - beta2 ** (t + 1))
    next_x = x - learning_rate * mt / (np.sqrt(vt) + epsilon)
    
    config['m'] = m
    config['v'] = v
    config['t'] = t + 1
    return next_x, config
