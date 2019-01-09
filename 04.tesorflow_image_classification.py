#!/usr/bin/env python

#call1
#%matplotlib inline
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("data/MNIST",one_hot = True)
print("size of::")
print("-Traning-set:\t\t{}".format(len(data.train.labels)))
print("-Test-set:\t\t{}".format(len(data.test.labels)))
print("-Validation-set:\t{}".format(len(data.validation.labels)))

data.test.labels[0:5,:]
data.test.cls = np.array([label.argmax() for label in data.test.labels])

data.test.cls


#we know thw MNIST image are 28 pixel in each dimension
image_size = 28

#image are store in one-dimension arrays of this length
image_size_flat = image_size * image_size

#Tuple with hight and width of image used to reshape array
image_shape = (image_size,image_size)

#number of classes, one class for each of 10 digits
num_classes = 10

#3x3 MNIST image view

def plot_images(image,cls_true,cls_pred = None):
    assert len(image) == len(cls_true) == 9

    #create figure with 3x3 sub-plots
    fig,axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)

    for i,ax in enumerate(axes.flat):
        #plot image
        ax.imshow(image[i].reshape(image_shape),cmap='binary')

        #show the true predection classes
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0},pred: {1}".format(cls_true[i],cls_pred[i])

        ax.set_xlabel(xlabel)

        #remove tricks from plot
        ax.set_xticks([])
        ax.set_yticks([])


#get the first image from the test-set
images = data.test.images[0:9]

#get the true classes for those images
cls_true = data.test.cls[0:9]

#ploat he images and labels using our helper-function above
plot_images(image=images,cls_true=cls_true)
