from rbm import RestrictedBoltzmannMachine as RBM, Joint as jRBM
from data import *
from utils import tile_raster_images
import os
import math
import numpy as np
from time import time
# if python < 3.4  -> pip install enum34
#from enum import Enum
#sudo pip install Pillow
try:
    import PIL.Image as Image
except ImportError:
    import Image

#Type control with enumeration




"""
    Runs a test for a given RBM model, dataset (classification problem), 
    and specified hyperparameters:
    :param    model: model of RBM to use: 'generative', 'joint' or 'discriminative'
    :param    data: specifies on which dataset to run a test: 'MNIST', 'CIFAR'
    :param    dFormat: specifies source data format 'pkl' or 'csv'
    :param    train_size: specifies number of samples from dataset to train 
    :param    test_size: specifies number of samples from dataset to test
    :param    binary: if binary problem is solved 
    :param    binarizationThreshold - a threshold for binarization of data
    :param    batch size: size of minibatch 
    :param    lr: learning rate
    :param    scal: sampling from Gaussian distribution with mean=0 and std var = scal
    :param    nrHiddenUnits: number of hidden units
    :param    nrEpochs_p: number of training epochs
    :param    nrOfIter: number of iterations for sampling
    :param    randomState: random state used for random nr generating
    :param    errorThreshold: denotes stop condition for RBM training
    :param    momentum_p
    :param    CDk_p: number of Gibbs step used by contrastive divergence 
        ...
"""       

def runTest(
            model = 'joint',
            data = 'MNIST',
            dFormat = 'pkl',
            train_size = 50,
            test_size = 10,
            binary = True,
            binarizationThreshold = 0.5,
            batch_size = 10,
            lr = 0.05,
            scal = 0.01,
            nrHiddenUnits_p = 700,
            nrEpochs_p = 100,
            nrOfIter = 1000,
            randomState = 1234,
            errorThreshold = 5,
            momentum_p = 0.0,
            CDk_p=1):
    
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
                trainX = binarize(trainX, threshold = binarizationThreshold)
                testX= binarize(trainX, threshold = binarizationThreshold)
                validX= binarize(validX, threshold = binarizationThreshold)
            trainX = trainX[:train_size]
            trainY = trainY[:train_size]
            testX = testX[:test_size]
            testY = testY[:test_size]
            validX = validX[:test_size]
            validY = validY[:test_size]
        else:
            #Case of reading from CSV
            if dFormat =='csv':
                dataSet = dataObj.readCSVDataFast(s=train_size)
                trainY = dataSet[:train_size,0]
                if binary:
                    trainX = binarize(scale(dataSet[:train_size,1:]), 
                                      threshold = binarizationThreshold)
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
        RBM1 = RBM(nrVisibleUnits, nrHiddenUnits_p, scal = scal, binary=binary, 
                   rnGen=rnGen)
        t1 = time() 
        RBM1.train(trainX, trainY, 5, lr, errorThreshold)
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
        #trainSet = np.concatenate([trainX,trainY], axis=1)
        #print trainSet.shape
        nrVisibleUnits = len(trainX[1])
        nrTargetUnits = len(trainY[1])
        #print nrVisibleUnits
        #Nr of visible units is 784 + 10 = 794
        #split the training set into mini-batches of batch_size
        numOfBatches = trainX.shape[0] / batch_size
        #print numOfBatches
        #initialize jRBM
        jRBM1 = jRBM(nrVisibleUnits, nrHiddenUnits_p, nrTargetUnits, 
                     scal = scal, binary=binary, rnGen=rnGen)
        epoch = 0
        mErrorX = 0
        mErrorY = 0
        mEs = np.zeros(nrEpochs_p+1)
        #for each epoch 
        t1 = time()
        while epoch < nrEpochs_p:
            t2 = time()
            epoch += 1
        #perform training based on CD for each minibatch
            for i in range(numOfBatches):
                #iterate to next batch
                #print i*batch_size
                batchX = trainX[i*batch_size:((i*batch_size)+batch_size)]
                batchY = trainY[i*batch_size:((i*batch_size)+batch_size)]
                #perform train on this batch and update weights globally
                gWVH, gWTH, gV, gT, gH, eX, eY = jRBM1.train(batchX, batchY, errorThreshold, k=CDk_p )
                jRBM1.updateWeight(lr, gWVH, gWTH, gV, gT, gH ,momentum = momentum_p)
                mErrorX += eX
                mErrorY += eY
            train_time = time()-t2
            #print mean error on data and labels
            mErrorX /= numOfBatches
            mEs[epoch] = mErrorX
            mErrorY /= numOfBatches
            print "MSE for epoch %d: on data: %0.3f, epoch time: %0.2f seconds" \
            % (epoch, mErrorX, time()-t2)
        train_time =  time()-t1
        print "Train time: %0.3fs" % train_time
        
        #Use return if plotResult function is called
        #return mEs
        """
        #Plot mean squared error on data within epochs
        plt.figure()
        plt.title('Mean squared error for epochs')
        plt.plot(mEs, 'b', label='Model with learning rate=%.4f' % lr)
        plt.legend()
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Mean-squared weight error')
        plt.xlim(xmin=1)
        plt.show()
       
        # Plot filters after training 
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=jRBM1.WeightsVH.T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.show()

        #Do sampling for one case of unseen data
        #dataObj.plot(validX[0])
        print "Original label: %d" % validY[0]
        #dataObj.plot(validX[2])
        print "Original label: %d" % validY[2]
        #validY = dataObj.transformLabel(validY)
        t2=time()
        reconstructedY = jRBM1.sample(validX[0], nrOfIter)
        print reconstructedY
        #print reconstructedY.argmax(axis=0)
        sample_time = time() - t2
        print("Sampling time: %0.3fs" % sample_time) 
        #dataObj.plot(reconstructedX)
        label = dataObj.inverseTransformLabel(reconstructedY)
        #label = reconstructedY.argmax(axis=0)
        print "Reconstructed label: %d" % label
        #print "Reconstructed label:"
        
        #for i in range(len(reconstructedY)):
            #if reconstructedY[i] == 1:
                #print i
       
        reconstructedY = jRBM1.sample(validX[2], nrOfIter)
        print "Reconstructed label:"
        print reconstructedY
        label = dataObj.inverseTransformLabel(reconstructedY)
        #print label
        print "Reconstructed label: %d" % label
        #for i in range(len(reconstructedY)):
            #if reconstructedY[i] == 1:
                #print i
        """  
        #Compute classification error
        #lb2, validY = dataObj.transformLabel(validY)
        #for i in range(len(validX[0:3])):
            #dataObj.plot(validX[i])
        t3=time()
        label = jRBM1.predict(validX,numOfIteration=nrOfIter)
        predict_time = time()-t3
        #count how many had wrong predicted label
        #also is wrong if more than one classes are predicted
        #trainY = dataObj.
        #trainY = dataObj.inverseTransformLabel(trainY, set=True)
        #print trainY
        counter = 0
        for i in range(len(label)):
            print "Reconstrlabel is %f, original label is%f" % (label[i],validY[i]) 
            if label[i] != validY[i]:
                counter +=1
        acc = 1 - (counter / float(len(label)))
        print counter / float(len(label))
        print counter
        print "Accuracy is %0.3f" % acc
        print "Prediction time is %0.3fs" % predict_time

#Simple test to check RBMs initialization and inheritance
def simpleTest():
    #Test on basic RBM model
    RBM1 = RBM(numOfVisibleUnits = 5, numOfHiddenUnits=2, weights=[], rnGen = 5234, 
                 scal=0.01, binary=True)
    print RBM1.Weights
    
    # Test on inherited class
    jRBM1 = jRBM(numOfVisibleUnits = 5, numOfHiddenUnits=2, numOfTargetUnits = 2,
                 rnGen = 7234, scal=0.01, binary=True)
    print jRBM1.NumOfVisibleUnits
    print jRBM1.Weights

# miscellaneous short tests   
def miscTest():
    #Checking current working directory and relative path 
    #print os.getcwd()
    #print os.path.relpath('C:\Users\Katarzyna Tarnowska\git\RBM-Classification\data\mnist_train.csv')
    print (RBMType.generative.name)
    

def plotResults(lr1 = 0.5,momentum = 0.5, 
                nrH1 = 700, nrH2 = 300,
                seed1 = 9999, batch_size = 5,
                CDk1=2, CDk2 = 3,
                nr_epochs = 100):
    
    plt.figure()
    plt.title('Convergence comparison of different models')
    mswe_base = runTest(nrEpochs_p = nr_epochs)
    plt.plot(mswe_base, 'k', label='Base model')
    mswe_lr1 = runTest(lr=lr1, nrEpochs_p = nr_epochs)
    plt.plot(mswe_lr1, 'b', label='Model with learning rate=%.1f' % lr1)
    mswe_mom1 = runTest(momentum_p = momentum, nrEpochs_p = nr_epochs)
    plt.plot(mswe_mom1, 'b--', label='Model with momentum=%.1f' % momentum)
    mswe_nrH1=runTest(nrHiddenUnits_p=nrH1, nrEpochs_p = nr_epochs)
    plt.plot(mswe_nrH1, 'g', label='Model with %d Hidden Units' % nrH1)
    mswe_nrH2=runTest(nrHiddenUnits_p=nrH2, nrEpochs_p = nr_epochs)
    plt.plot(mswe_nrH2, 'g--', label='Model with %d Hidden Units' % nrH2)
    mswe_nrS1 = runTest(randomState=seed1, nrEpochs_p = nr_epochs)
    plt.plot(mswe_nrS1, 'r', label='Model with random state =%d' % seed1)
    mswe_nrS2 = runTest(batch_size =batch_size , nrEpochs_p = nr_epochs)
    plt.plot(mswe_nrS2, 'r--', label='Model with batch size =%d' % batch_size)
    mswe_CDk1 = runTest(CDk_p = CDk1, nrEpochs_p = nr_epochs)
    plt.plot(mswe_CDk1, 'm', label='Model with %d-step contrastive divergence' % CDk1)
    mswe_CDk2 = runTest(CDk_p=CDk2, nrEpochs_p = nr_epochs)
    plt.plot(mswe_CDk2, 'm--', label='Model with %d-step contrastive divergence'  % CDk2 )

    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Mean-squared error on data')
    plt.xlim(xmin=1)
    plt.show()

runTest();
#simpleTest();
#miscTest();
#plotResults()

