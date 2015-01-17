from rbm import RestrictedBoltzmannMachine as RBM, Discriminative as DRBM
from data import *
import math
import numpy as np
from time import time

"""
    Run a test for a given dataset (classification problem)
    and specified hyperparameters:
    :param    binary: if data is in binary format
    :param    lr: learning rate
    :param    scal: sampling from Gaussian distribution of 0 and scal
    :param    nrHiddenUnits: number of hidden units
    :param    nrOfIter: number of iterations
    :param    batch size: size of minibatch
    :param    randomState: random state used for random nr generating
    :param    errorThreshold: denotes stop condition for RBM training
        ...
"""       

def runTest(data = 'MNIST',
            dFormat = 'pkl',
            binary = True,
            lr = 0.05,
            scal = 0.01,
            nrHiddenUnits_p = 400,
            nrOfIter = 1000,
            batch_size = 50000,
            randomState = 1234,
            errorThreshold = 5):
    
    t0=time()
    # Case for MNIST data
    if (data == 'MNIST'):
        dataObj = MNIST();
        # Case of reading from cPickle format
        if dFormat == 'pkl':
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
            if dFormat =='CSV':
                dataSet = dataObj.readCSVDataFast(s=batch_size)
                trainY = dataSet[:batch_size,0]
                if binary:
                    trainX = binarize(scale(dataSet[:batch_size,1:]))
                else:
                    trainX = scale(dataSet[:batch_size,1:])
    read_time = time() - t0
    
    
    nrVisibleUnits = len(trainX[1]);
    rnGen = np.random.RandomState(randomState)
    #dataObj.plot(trainX[3])
    
    RBM1 = RBM(nrVisibleUnits, nrHiddenUnits_p, scal = scal, binary=binary, rnGen=rnGen)
    t1 = time() 
    RBM1.train(trainX, trainY, 1, lr,errorThreshold)
    train_time = time() - t1
    print("Train time: %0.3fs" % train_time) 
        
    t2=time()
    dataObj.plot(RBM1.sample(nrOfIter))
    sample_time = time() - t2
    print("Sampling time: %0.3fs" % sample_time) 
    print("Read data time: %0.3fs" % read_time) 
    #RBM1.sample(100)  
    #dataObj.visualize(RBM1.sample(100))
    #MNIST.visualize(trainingData[0,:])
      
#Check RBMs initialization
def simpleTest():
    #RBM1 = RBM(5,2)
    #print RBM1.Weights
    
    DRBM1 = DRBM(5,2,2)
    print DRBM1.NumOfVisibleUnits
    #print DRBM1.Weights
    
def dataFunTest(trainY):
    print MNIST().transformLabel(trainY)[0]
    
runTest()
#simpleTest()
