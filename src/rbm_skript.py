from rbm import RestrictedBoltzmannMachine as RBM, Joint as jRBM
from data import *
import os
import math
import numpy as np
from time import time

"""
    Runs a test for a given RBM model, dataset (classification problem), 
    and specified hyperparameters:
    :param    model: model of RBM to use: 'generative', 'joint' or 'discriminative'
    :param    data: specifies on which dataset to run a test: 'MNIST', 'CIFAR'
    :param    dFormat: specifies source data format 'pkl' or 'csv'
    :param    train_size: specifies number of samples from dataset to train 
    :param    binary: if binary problem is solved 
    :param    batch size: size of minibatch 
    :param    lr: learning rate
    :param    scal: sampling from Gaussian distribution of 0 and scal
    :param    nrHiddenUnits: number of hidden units
    :param    nrOfIter: number of iterations for sampling
    :param    randomState: random state used for random nr generating
    :param    errorThreshold: denotes stop condition for RBM training
        ...
"""       

def runTest(
            model = 'joint',
            data = 'MNIST',
            dFormat = 'pkl',
            train_size = 50000,
            binary = True,
            batch_size = 10,
            lr = 0.05,
            scal = 0.01,
            nrHiddenUnits_p = 400,
            nrOfIter = 1000,
            randomState = 1234,
            errorThreshold = 5):
    
    # Generate random state
    rnGen = np.random.RandomState(randomState)
    
    # Read specified dataset from source files
    t0=time()
    # Case for MNIST data
    if (data == 'MNIST'):
        dataObj = MNIST();
        # Case of reading from cPickle format
        if dFormat == 'pkl':
            trainingSet, validationSet, testSet = dataObj.loadData()
            trainX, trainY = trainingSet
            validX, validY = validationSet
            testX, testY = testSet
            #case of binary data
            if binary:
                trainX = binarize(trainX)
            trainX = trainX[:train_size]
        else:
            #Case of reading from CSV
            if dFormat =='csv':
                dataSet = dataObj.readCSVDataFast(s=train_size)
                trainY = dataSet[:train_size,0]
                if binary:
                    trainX = binarize(scale(dataSet[:train_size,1:]))
                else:
                    trainX = scale(dataSet[:train_size,1:])
    # Case for CIFAR data
    if (data == 'CIFAR'):
        pass
        #TO DO
    read_time = time() - t0
    
    #dataObj.plot(trainX[3])
    #dataObj.visualize(trainX[0,:])
    
    #Initialize a chosen RBM model and perform train and sample operations
    if(model == 'generative'): 
        nrVisibleUnits = len(trainX[1]);
        RBM1 = RBM(nrVisibleUnits, nrHiddenUnits_p, scal = scal, binary=binary, rnGen=rnGen)
        t1 = time() 
        RBM1.train(trainX, trainY, 1, lr, errorThreshold)
        train_time = time() - t1
        print("Read data time: %0.3fs" % read_time)
        print("Train time: %0.3fs" % train_time) 
            
        t2=time()
        example = RBM1.sample(nrOfIter)
        sample_time = time() - t2
        print("Sampling time: %0.3fs" % sample_time) 
        dataObj.plot(example)
        #dataObj.visualize(RBM1.sample(nrOfIter))
    
    if(model == 'joint'):
        #transform training labels using LabelBinarization to model joint probabilities
        trainY = dataObj.transformLabel(trainY)
        #print (trainY)[0]
        #concatenate trainX and binarized trainY into one np array
        trainSet = np.concatenate([trainX,trainY], axis=1)
        #print trainSet.shape
        nrVisibleUnits = len(trainSet[1])
        #print nrVisibleUnits
        #Nr of visible units is 784 + 10 = 794
        #split the training set into mini-batches of batch_size
        numOfBatches = trainSet.shape[0] / batch_size
        print numOfBatches
        #initialize jRBM
        jRBM1 = jRBM(nrVisibleUnits, nrHiddenUnits_p, scal = scal, binary=binary, rnGen=rnGen)
        #perform training based on CD for each minibatch
        for i in range(numOfBatches):
            #iterate to next batch
            #print i*batch_size
            batch = trainSet[i*batch_size:((i*batch_size)+batch_size)]
            #perform train on this batch
            jRBM1.train(batch, lr, errorThreshold)
          
#Simple test to check RBMs initialization and inheritance
def simpleTest():
    #Test on basic RBM model
    RBM1 = RBM(numOfVisibleUnits = 5, numOfHiddenUnits=2, weights=[], rnGen = 5234, 
                 scal=0.01, binary=True)
    print RBM1.Weights
    
    # Test on inherited class
    jRBM1 = jRBM(numOfVisibleUnits = 5, numOfHiddenUnits=2, weights=[], rnGen = 7234, 
                 scal=0.01, binary=True)
    print jRBM1.NumOfVisibleUnits
    print jRBM1.Weights

# miscellaneous short tests   
def miscTest():
    #Checking current working directory and relative path 
    print os.getcwd()
    print os.path.relpath('C:\Users\Katarzyna Tarnowska\git\RBM-Classification\data\mnist_train.csv')
    
runTest();
#simpleTest();
#miscTest();
