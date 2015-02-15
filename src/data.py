# Module for loading and preprocessing different data

import sys
import numpy as np
import cPickle
import gzip
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class MNIST(object):
    def __init__(self):
        self.labelBin = None
# Reads CSV file from a given path and returns numpy arrays of n data with or without labels
    def readCSVData(self, datafile ='data/mnist_train.csv', labels=False, n=-1):
        A = np.genfromtxt(datafile, delimiter = ",", dtype = "uint8")
        print(A.shape)
        if labels:
            if (n>-1):
                label = A[:n, 0]
                trainingData = A[:n, 1:]
            else:
                label = A[:, 0]
                trainingData = A[:, 1:]
            return (label, trainingData)  
        else:
            if (n>-1):
                trainingData = A[:n] 
            else:
                trainingData = A[:] 
            return trainingData
    
    # fast reading data method for csv-file with size s x f
    def readCSVDataFast(self, datafile = 'data\mnist_train.csv', s=10000, f=785):
        test_cases = open(datafile, 'r')
        counter = 0
        trainingData = np.zeros((s,f))
        for test in test_cases:
            while counter < s:
                trainingData[counter,:] = np.array([int(i) for i in test.split(",")]) 
                counter += 1
        test_cases.close()
        return trainingData
    
    """
    Load MNIST data from pkl file
    Returns a tuple containing the training data, validation data, test data
    
    Training data is tuple of two entries
        numpy ndarray with 50 000 entries, each entry is, a
        numpy ndarray with 784 values (28 * 28 = 784 pixels in a single image)
       AND
        numpy ndarray of 50 000 entries - the digit values (0...9)
        
    Validation and test data have the same structure, excpet they contain
    10 000 entries
    """
    def loadData(self, datafile = 'data/mnist.pkl.gz'):
        fil = gzip.open(datafile, 'rb')
        trainingSet, validationSet, testSet = cPickle.load(fil)
        fil.close()
        return (trainingSet, validationSet, testSet)
    
    """Transforms 1-column numpy array digits 0-9 into 
    10-column np arry of binary values
    uses sklearn LabelBinarizer facility
    Suitable for joint probability learning binary"""
    def transformLabel(self, trainY):
        self.labelBin = LabelBinarizer()
        self.labelBin.fit(trainY)
        return self.labelBin.transform(trainY)
    
    """Transforms 10-column np arry of binary values into 1-column numpy array digits 0-9 
    uses sklearn LabelBinarizer facility
    takes LabelBinarizer fitted before as input"""
    def inverseTransformLabel(self, binTrainY, set=False):
        #return self.labelBin.inverse_transform(binTrainY, threshold = 0)
        if set:
            return binTrainY.argmax(axis=1)
        else:
            return binTrainY.argmax(axis=0)
        
    
    """ Loads data using sklearn package fech function
    Fetch an mldata.org data set
    Returns Dictionary-like object, 
    the interesting attributes are: data, the data to learn, 
    target, the classification labels"""
    def loadMNIST(self):
        mnist = fetch_mldata('MNIST original')
        trainingData, label = mnist.data, mnist.target
        X = np.asarray(trainingData, 'float32')
        return (label, X)
    
    """ Loads data from the CIFAR data set which can be downloaded under
    http://www.cs.toronto.edu/~kriz/cifar.html
    This data set consists of 60000 32x32 RGB images which belong
    to one of 10 classes
    Returns a dictonary containing the data and the labels
    """
    def loadCIFAR(self,file):
	    fo = open(file, 'rb')
	    dict = cPickle.load(fo)
	    fo.close()
	    return dict

    #Print data from np array
    def printData(self,td):
        print 'MNIST data:', td
    
    #Generates set few times bigger by adding similar data 
    def nudge_dataset(self,X, Y):
        # TO DO optionally
        return (X,Y)
    
    #Shows an image of the MNIST dataset given as a binary vector 
    def visualize(self, example):
        counter = 0
        for i in range(28):
            for j in range(28):
                print int(example[counter]),
                counter += 1
            print "\n"
    
    #Plots an image of the MNIST dataset given as a binary vector 
    def plot(self, example):
        plt.imshow(example.reshape((28, 28)), cmap = 'Greys')
        plt.show()
    
    #Plots multiple images of MNIsT dataset in a 10x10 table    
    def plot10x10(self, examples):
        pass

#Transform data into binary (black and white images) using threshold method
def binarize(X, threshold = 0.5):
    # Convert image array to binary with threshold 
    X = X > threshold
    return X
    
#Transform data into floats from [0.0 ; 1.0]
def scale(X):    
    X = np.asarray( X, 'float32')
    data = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    return data

#Saves numpy array to csv file
def saveData(outfile, data):
    np.savetxt(outfile, data, delimiter=",")

# Plots a chart given np array of mswe over iterations for different models    
def plotResults(mswe, model_selection):
    mswe_lr1, mswe_lr2, mswe_nrH1, mswe_nrH2, mswe_nrS1, mswe_nrS2 = mswe
    lr1,lr2, nrH1, nrH2,nrS1,nrS2   = model_selection
    plt.figure()
    plt.title('Convergence comparison of different models')
    plt.plot(mswe_lr1, 'b', label='Model with learning rate=%.4f' % lr1)
    plt.plot(mswe_lr2, 'b--', label='Model with learning rate=%.4f' % lr2)
    plt.plot(mswe_nrH1, 'g', label='Model with nr Hidden Units=%.0f' % nrH1)
    plt.plot(mswe_nrH2, 'g--', label='Model with nr Hidden Units=%.0f' % nrH2)
    plt.plot(mswe_nrS1, 'r', label='Model with number of iterations for sampling =%.0f' % nrS1)
    plt.plot(mswe_nrS2, 'r--', label='Model with number of iterations for sampling =%.0f' % nrS2)
    plt.legend()
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel('Mean-squared weight error')
    plt.show()


""" Loads data from the CIFAR data set which can be downloaded under
http://www.cs.toronto.edu/~kriz/cifar.html
This data set consists of 60000 32x32 RGB images which belong
to one of 10 classes
Returns a dictonary containing the data and the labels
"""
def loadCIFAR(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#Plots an image of the MNIST dataset given as a binary vector 
def plotCIFAR(example):
    plt.imshow(example.reshape(32,32), cmap = 'greys')
    plt.show()

#Plots an image of the CIFAR dataset given as a binary vector 
def plotCIFAR(example):
	A = np.zeros((1024,3))
	for i in range(1024):
		A[i,:] = [example[i],example[i+1024],example[i+2048]]
	
    plt.imshow(A.reshape(32,32,3))
    plt.show()
