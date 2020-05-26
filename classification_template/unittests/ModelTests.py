#!/usr/bin/env python
"""
model tests
"""

import os
import sys
import unittest

## import model specific functions and variables
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(test=True)
        self.assertTrue(os.path.exists(SAVED_MODEL))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        model = model_load()
        
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        model = model_load()
    
        ## ensure that a list can be passed
        query = [6.1,2.8]
        
        result = model_predict(query,model,test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred[0] in [0,1,2])

    def test_04_predict(self):
        """
        test the predict function accuracy
        """

        ## load model first
        model = model_load()
        
        ## example predict
        example_queries = [np.array([[6.1,2.8]]), np.array([[7.7,2.5]]), np.array([[5.8,3.8]])]
        
        for query in example_queries:
            result = model_predict(query,model,test=True)
            y_pred = result['y_pred']
            self.assertTrue(y_pred in [0,1,2])

    def test_05_predict(self):
        """
        ensure that we can predict in batches
        """

        ## load model first
        model = model_load()
        
        ## example predict
        query = np.array([[6.1,2.8],[7.7,2.5],[5.8,3.8]])

        results = model_predict(query,model,test=True)
        y_preds = results['y_pred']
        for y_pred in y_preds:
            self.assertTrue(y_pred in [0,1,2]) 
            
### Run the tests
if __name__ == '__main__':
    unittest.main()
