#tutorial: https://www.youtube.com/watch?v=t0EzVCvQjGE
import cv2
import numpy
import matplotlib.pyplot
#tensorflow - Neural Networks
from tensorflow.keras import datasets, layers, models

#saving dataset in same tuple format
(trainingImages, trainingLabels) , (testingImages, testingLabels) = datasets.cifar10.load_data()
