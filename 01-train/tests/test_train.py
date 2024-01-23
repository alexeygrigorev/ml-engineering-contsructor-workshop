import os
import unittest

from datetime import datetime

from duration_prediction.train import train

class TestMLPipeline(unittest.TestCase):

    def test_train_function(self):
        # Test if train function works and saves the model file
        
        train_month = datetime(2022, 1, 1)
        val_month = datetime(2022, 2, 1)
        model_output_path = 'test_model.bin'
        
        train(train_month, val_month, model_output_path)

        # Check if model file is created
        self.assertTrue(os.path.exists(model_output_path))

        # Remove test model file after test
        os.remove(model_output_path)

if __name__ == '__main__':
    unittest.main()