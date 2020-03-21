import numpy as np

class KNN :
    # Constructor
    def __init__(self, training_images, training_labels) :
        self.training_images = training_images
        self.training_labels = training_labels
    
    def predict(self, test_images) :
        # Some Code
        return test_labels