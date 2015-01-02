
# Loading and preprocessing data

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import binarize

# Reads CSV file from a given path and returns numpy arrays of n data with or without labels
def readData(datafile, labels=False, n=-1):
    A = np.genfromtxt(datafile, delimiter = ",", dtype = "uint8")
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

# Load MNIST data
def loadMNIST():
    mnist = fetch_mldata('MNIST original')
    trainingData, label = mnist.data, mnist.target
    return (label, trainingData)

#Print data from np array
def printData(td):
    print 'MNIST data:', td
    
#Transform data into binary using threshold method
def binarizeH(X):
    # Convert image array to binary with threshold 
    return binarize(X, 127)

#Transform data into floats from [0.0 ; 1.0]
def scale(X):    
    X = np.asarray( X, 'float32')
    trainingData = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    return trainingData

#Generates set few times bigger by adding similar data 
def nudge_dataset(X, Y):
    # TO DO optionally
    return (X,Y)

#Plots images of MNIST data 
def visualize(X):
    #TO DO
    return X

#Saves numpy array to csv file
def saveData(outfile, data):
    np.savetxt(outfile, data, delimiter=",")