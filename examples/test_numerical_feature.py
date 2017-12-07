# -*- coding: utf-8 -*-

import numpy as np
import unittest
import copy
import numbers
from scipy.sparse import csc_matrix, csr_matrix

import sys
sys.path.append('../')
from ml.feature import *

class TestNumericalFeature(unittest.TestCase):
    def setUp(self):
        self.dense_values1 = [0, 4, 5.5, 9, 3.7, 0, 1, 0, 0, 0, 0]
        self.df1 = NumericalFeature(self.dense_values1, name='df1')
        self.sparse_values1 = csc_matrix(self.dense_values1)
        self.sparse_values1.eliminate_zeros()
        self.sf1 = NumericalFeature(self.sparse_values1, name='sf1')
        
        self.dense_values2 = np.array([1, 2, 3, 4, 5, 6, 7]).reshape((-1, 1))
        self.df2 = NumericalFeature(self.dense_values2, name='df2')
        self.sparse_values2 = csc_matrix(self.dense_values2)
        self.sparse_values2.eliminate_zeros()
        self.sf2 = NumericalFeature(self.sparse_values2, name='sf2')

    def test_init_shape(self):
        self.assertTrue(self.df1.shape == (len(self.dense_values1),))
        self.assertTrue(self.df2.shape == (self.dense_values2.shape[0],))
        self.assertTrue(self.sf1.shape == (1, len(self.dense_values1)))
        self.assertTrue(self.sf2.shape == (1, self.dense_values2.shape[0]))
        
    def test_init_array(self):
        self.assertTrue(isinstance(self.df1._values, np.ndarray))
        self.assertTrue(isinstance(self.df2._values, np.ndarray))
        self.assertTrue(isinstance(self.sf1._values, csr_matrix))
        self.assertTrue(isinstance(self.sf2._values, csr_matrix))
      
    def test_init_shape_error(self):
        with self.assertRaises(ValueError):
            feature = NumericalFeature(np.array(self.dense_values1)[:, None, None], name='f2')
            
    def test_init_numeric_error(self):
        feature = NumericalFeature(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), name='feature2')
        feature = NumericalFeature(np.array([1.4, 8.2, 82.2]), name='feature3')
        with self.assertRaises(TypeError):
            feature = NumericalFeature(np.array(list('jskdfjdskfjsd')), name='feature4')
            
    def test_categorical(self): # TODO
        values = [0.3, 0.5, 0.8, 0.2, 0.6, 0.1]
        name = 'feature2'
        feature = NumericalFeature(values, name)

if __name__ == '__main__':
    unittest.main()
