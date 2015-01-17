from rbm import RestrictedBoltzmannMachine as RBM, Discriminative as DRBM
from data import *
import math
import numpy as np

"""Run a test for a given dataset (classification problem)
    and specified hyperparameters:
        bin - if data is in binary format
        lr - learning rate
        scal - sampling from Gaussian distribution of 0 and scal
        nrHiddenUnits - number of hidden units
        nrOfIter - number of iterations
        batch size - size of minibatch
        ...
"""       

def runTest(data = 'MNIST',
            read = 'pkl',
            binary = True,
            lr = 0.05,
            scal = 0.01,
            nrHiddenUnits = 100,
            nrOfIter = 10000,
            batch_size = 10000):
    
    # Case for MNIST data
    if (data == 'MNIST'):
        dataObj = MNIST();
        # Case of reading from cPickle format
        if read == 'pkl':
            trainingSet, validationSet, testSet = dataObj.loadData()
            trainX, trainY = trainingSet
            #validX, validY = validationSet
            #testX, testY = testSet
            #case of binary data
            if binary:
                trainX = binarize(trainX)
            trainX = trainX[:batch_size]
        else:
            #Case of reading from CSV
            if read =='CSV':
                dataSet = dataObj.readCSVDataFast(s=batch_size)
                trainY = dataSet[:batch_size,0]
                if binary:
                    trainX = binarize(scale(dataSet[:batch_size,1:]))
                else:
                    trainX = scale(dataSet[:batch_size,1:])
        nrVisibleUnits = len(trainX[1]);
        
        dataObj.plot(trainX);
        #RBM1 = RBM(nrVisibleUnits, nrHiddenUnits, scal = scal, bin=bin)
        #RBM1.train(trainX, trainY, 5, lr,5)
        #dataObj.visualize(RBM1.sample(100))
        #MNIST.visualize(trainingData[0,:])

#Check RBMs initialization
def simpleTest():
    RBM1 = RBM(5,2)
    print RBM1.Weights
    
    DRBM1 = DRBM(5,2,2)
    print DRBM1.Weights
    
runTest()
#simpleTest()
