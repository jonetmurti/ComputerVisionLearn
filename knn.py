import numpy as np
from statistics import mode

class KNN :
    # Constructor
    def __init__(self, training_images, training_labels, k_value) :
        self.training_images = training_images
        self.training_labels = training_labels
        self.k_value = k_value

    def euclidDist(matA, matB) :
        # Numpy Matrix or Matrices with same sizes as input
        dist = np.sqrt(np.sum((np.subtract(matA, matB))**2, axis=1))
        return dist
    
    def predict(self, test_images) :
        test_labels = []
        for test_image in test_images :
            distances = euclidDist(test_image, self.training_images)
            dist_sort = np.argsort(distances)
            test_labels.append(statistics.mode(this.training_labels[dist_sort[:this.k_value:1]]))
            
        return test_labels