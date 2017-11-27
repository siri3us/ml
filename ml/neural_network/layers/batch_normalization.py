# -*- coding: utf-8 -*-
import random
import numpy as np
from ..sequential import Layer

"""class BatchMeanSubtraction(Layer):
    def __init__(self, alpha = 0.):
        super().__init__()
        self.alpha = alpha
        self.old_mean = None 
        
    def update_output(self, X_input):
        batch_mean = np.mean(X_input, axis=0)
        if self.old_mean is None:
            self.old_mean = batch_mean
        return self.output
    
    def update_input(self, input, grad_output):
        # Your code goes here. ################################################
        return self.grad_input
    
    def __repr__(self):
        return "BatchMeanNormalization"""
