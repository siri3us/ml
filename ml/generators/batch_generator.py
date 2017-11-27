# -*- coding: utf-8 -*-

import numpy as np
import random

class BatchGenerator:
    # TODO inherit from Checker
    def __init__(self, *arrays, batch_size=128, 
                 shuffle=True, infinite=False, fixed_batch_size=False,
                 random_state=0):
        """
        Arguments:
            arrays           - 
            batch_size       - batch size
            shuffle          - if True, data is shuffled
            infinite         - if True, generation never ends (no StopIteration rised)
            fixed_batch_size - if True, only batches of batch_size samples are returned 
            random_state     - random state
        """
        # Setting and checking arrays
        self.arrays = list(arrays)
        assert len(self.arrays) > 0, 'At least 1 array must be provided.'
        sizes = []
        for n_array, array in enumerate(self.arrays):
            assert isinstance(array, np.ndarray), 'arrays[{}] must be a numpy array.'.format(n_array)
            sizes.append(array.shape[0])
        assert np.min(sizes) == np.max(sizes)
        self.n_samples = sizes[0]
        
        self.batch_size = batch_size
        assert self.batch_size <= self.n_samples, "batch_size {} must not exceed number of samples {}.".format(
            batch_size, self.n_samples)
        self.shuffle    = shuffle
        self.infinite   = infinite
        self.fixed_batch_size = fixed_batch_size
        self.random_state = random_state
        self.random_gen = random.Random(random_state)
        
    def restart(self):
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            self.random_gen.shuffle(self.indices)
        self.curr_pos = 0
        
    def __iter__(self):
        self.restart()
        return self
    
    def __next__(self):
        if self.curr_pos >= self.n_samples:
            raise StopIteration

        batches = []
        self.next_pos = min(self.curr_pos + self.batch_size, self.n_samples)
        for array in self.arrays:
            batches.append(array[self.indices[self.curr_pos : self.next_pos]])
        self.curr_pos = self.next_pos

        next_batch_size = min(self.curr_pos + self.batch_size, self.n_samples) - self.curr_pos  
        if self.fixed_batch_size:
            if next_batch_size < self.batch_size:
                if self.infinite:
                    self.restart()                 # Next __next__ starts with new indices
                else:
                    self.curr_pos = self.n_samples # Next __next__ raises StopIteration
        else:
            if self.infinite:
                if next_batch_size == 0:
                    self.restart()
                    
        if len(batches) == 1:
            return batches[0]
        return batches
