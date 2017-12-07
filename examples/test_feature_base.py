# -*- coding: utf-8 -*-

import numpy as np
import unittest
import numbers
import copy
from scipy.sparse import csr_matrix, csc_matrix
from collections import Counter
    
import sys
sys.path.append('../')
from ml.feature import FeatureBase
        
        
class TestFeatureBase(unittest.TestCase):
    def setUp(self):
        self.values1 = [0, 1, 0, 0, 0, 1, 1.1, 1, 0, 4, 1, 7, 0, 0, 0]
        
    def test_init(self):
        self.dense_values = np.array(self.values1)     # (L, )
        
        self.sparse_values1 = csc_matrix(self.values1) # (1, L)
        self.sparse_values1.eliminate_zeros()          # (1, L)
        
        self.sparse_values2 = csc_matrix(np.array(self.values1)[:, np.newaxis]) # (L, 1)
        self.sparse_values2.eliminate_zeros()          # (L, 1)
        
        self.sparse_values3 = csr_matrix(self.values1) # (1, L)
        self.sparse_values3.eliminate_zeros()          # (1, L)
        
        self.sparse_values4 = csr_matrix(np.array(self.values1)[:, np.newaxis]) # (1, L)
        self.sparse_values4.eliminate_zeros()          # (1, L)
        
        feature  = FeatureBase(self.dense_values, name='f')
        feature1 = FeatureBase(self.sparse_values1, name='f1')
        feature2 = FeatureBase(self.sparse_values2, name='f2')
        feature3 = FeatureBase(self.sparse_values3, name='f3')
        feature4 = FeatureBase(self.sparse_values4, name='f4')
        
        self.assertTrue(feature.shape == (15,))
        self.assertTrue(feature1.shape == (1, 15))
        self.assertTrue(feature2.shape == (1, 15))
        self.assertTrue(feature3.shape == (1, 15))
        self.assertTrue(feature4.shape == (1, 15))
        
        self.assertTrue(isinstance(feature._values, np.ndarray))
        self.assertTrue(isinstance(feature1._values, csr_matrix))
        self.assertTrue(isinstance(feature2._values, csr_matrix))
        self.assertTrue(isinstance(feature3._values, csr_matrix))
        self.assertTrue(isinstance(feature4._values, csr_matrix))
        
        self.assertTrue(isinstance(feature.values, np.ndarray))
        self.assertTrue(isinstance(feature1.values, csc_matrix))
        self.assertTrue(isinstance(feature2.values, csc_matrix))
        self.assertTrue(isinstance(feature3.values, csc_matrix))
        self.assertTrue(isinstance(feature4.values, csc_matrix))
        
        self.assertEqual(feature.name, 'f')
        self.assertEqual(feature1.name, 'f1')
        self.assertEqual(feature2.name, 'f2')
        self.assertEqual(feature3.name, 'f3')
        self.assertEqual(feature4.name, 'f4')
        
        
        self.assertEqual(len(feature), 15)
        self.assertEqual(len(feature1), 15)
        self.assertEqual(len(feature2), 15)
        self.assertEqual(len(feature3), 15)
        self.assertEqual(len(feature4), 15)

    def test_get_values(self):
        self.dense_values = np.array(self.values1)
        self.sparse_values1 = csc_matrix(self.values1)
        self.sparse_values1.eliminate_zeros()
        self.sparse_values2 = csc_matrix(np.array(self.values1)[:, np.newaxis])
        self.sparse_values2.eliminate_zeros()
        self.sparse_values3 = csr_matrix(self.values1)
        self.sparse_values3.eliminate_zeros()
        self.sparse_values4 = csr_matrix(np.array(self.values1)[:, np.newaxis])
        self.sparse_values4.eliminate_zeros()
        
        feature = FeatureBase(self.dense_values, name='f')
        feature1 = FeatureBase(self.sparse_values1, name='f1')
        feature2 = FeatureBase(self.sparse_values2, name='f2')
        feature3 = FeatureBase(self.sparse_values3, name='f3')
        feature4 = FeatureBase(self.sparse_values4, name='f4')
        
        features = [feature, feature1, feature2, feature3, feature4]
        for sparse in [False, True]:
            for feature in features:
                returned_values = feature.get_values(sparse=sparse)
                if sparse:
                    self.assertTrue(isinstance(returned_values, csc_matrix))
                    self.assertTrue((returned_values.shape[0] > 1) & (returned_values.shape[1] == 1))
                    returned_values = returned_values.toarray()
                self.assertTrue(np.allclose(self.dense_values[:, np.newaxis], returned_values))
                
    def test_check_shaped(self):
        with self.assertRaises(ValueError):
            feature = FeatureBase(np.array(self.values1)[:, np.newaxis, np.newaxis], name='f', verbose=1)
      
    def test_check_constant(self):
        values = np.array([1, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            feature = FeatureBase(values, name='feature')
            feature._kernel._check_constant(feature._values)
             
        values = csc_matrix(np.array([2, 2, 2, 2, 2, 2])) 
        with self.assertRaises(ValueError):
            feature = FeatureBase(values, name='feature')
            feature._kernel._check_constant(feature._values)      
        

    def test_check_numeric(self):
        f_values = np.array([1, 'sd', 34, True, -1])
        with self.assertRaises(TypeError):
            feature_base = FeatureBase(f_values, name='feature')
            feature_base._check_numeric(feature_base._values)
            
if __name__ == '__main__':
    unittest.main()
