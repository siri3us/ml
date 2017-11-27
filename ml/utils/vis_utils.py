# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt, ceil
from itertools import product

def visualize_grid(Xs, ubound=255.0, padding=1, n_cols=5):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C): 
        N - number of images
        H - height
        W - width
        C - number of channels
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    n_rows = int(ceil(N / n_cols))
    
    grid_height = H * n_rows + padding * (n_rows - 1)
    grid_width  = W * n_cols + padding * (n_cols - 1)
    grid = np.zeros((grid_height, grid_width, C))
    
    next_idx = 0
    y0, y1 = 0, H
    for y in range(n_rows):
        x0, x1 = 0, W
        for x in range(n_cols):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * H + A, A * W + A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y, x in product(range(A), range(A)):
        if n < N:
            G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
            n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G
  
def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H,W,C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N*H+N, D*W+D, C), Xs.dtype)
    for y, x in product(range(N), range(D)):
        G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G
