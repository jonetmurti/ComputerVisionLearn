import numpy as np
import random
import operator
from statistics import mode

class KNN :
    # Constructor
    def __init__(self, training_images, training_labels, k_value) :
        self.training_images = training_images
        self.training_labels = training_labels
        self.k_value = k_value

    def euclidDist(self, matA, matB) :
        # Numpy Matrix or Matrices with same sizes as input
        dist = np.sqrt(np.sum((np.subtract(matA, matB))**2, axis=1))
        return dist
    
    def predict(self, test_images) :
        test_labels = list()
        for test_image in test_images :
            distances = self.euclidDist(test_image, self.training_images)
            dist_sort = np.argsort(distances)
            labels_sum = {}
            for key in dist_sort[:self.k_value:] :
              if self.training_labels[key] in labels_sum :
                labels_sum[self.training_labels[key]] += 1
              else :
                labels_sum[self.training_labels[key]] = 1
            
            max_key = max(labels_sum.items(), key=operator.itemgetter(1))[0]
            labels_candidate = []
            for key in labels_sum :
              if (labels_sum[key] == labels_sum[max_key]) :
                labels_candidate.append(key)
            
            test_labels.append(labels_candidate[random.randrange(0, len(labels_candidate), 1)])

            
            
        return test_labels


from tensorflow.keras.datasets import cifar10

(pict_train, label_train), (pict_test, label_test) = cifar10.load_data()
print("Number of training set :", pict_train.shape)
print("Number of label training set :", label_train.shape)
print("training pict_dimension :", pict_train.shape[0])
print("Number of test set :", pict_test.shape)
print("Number of label test set :", label_test.shape)
print("training test_dimension :", pict_test.shape[0])

flat_pict_train = np.array(pict_train).reshape(50000, 32*32*3)
flat_pict_test = np.array(pict_test).reshape(10000, 32*32*3)
flat_label_train = np.array(label_train).reshape(50000)
flat_label_test = np.array(label_test).reshape(10000)

print(flat_pict_train.shape)
print(flat_pict_test.shape)
print(flat_label_train.shape)
print(flat_label_test.shape)

model = KNN(flat_pict_train, flat_label_train, 3)
test_labels = model.predict(flat_pict_test)

print(test_labels)