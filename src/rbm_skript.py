#Main module for performing different tests
#for different RBM models
#and different datasets

#### Libraries
# Standard library
import os
from time import time
# if python < 3.4  -> pip install enum34
from enum import Enum

# Own library
from rbm import RestrictedBoltzmannMachine as RBM, Joint as jRBM, BinomialRestrictedBoltzmannMachine as rbmb
from data import *
from utils import tile_raster_images

# Third-party libraries
import math
import scipy
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
#sudo pip install Pillow
try:
    import PIL.Image as Image
except ImportError:
    import Image

#Type control with enumeration
class RBMType(Enum):
    generative = 'generative'  #model using data without labels
    joint =     'joint'        #models joint probabilities of data and labels
    discriminative =  'discriminative' # modification of joint - optimizes p(y|x), not p(x,y)
    binomial =  'binomial' #models binomial units

class Dataset(Enum):
    MNIST = 'MNIST'
    CIFAR = 'CIFAR'
    
class DataType(Enum):
    binary = 'binary' #{0,1},
    gaussian = 'gaussian' #<0;1> - for non-binary data 'normalization' is advised
    binomial = 'binomial' #any real value

class TerminationCondition(Enum):
    errThresh = 'errorThreshold'   #threshold for squared error
    epochNumber= 'epochNumber'        # number of iterations  

#global structure to store metrics across different test runs
results = [] 

"""
    Runs a test for a given RBM model, dataset (classification problem), 
    specified model hyperparameters and testrun parameters:
    dataset parameters
    :param    data: specifies on which dataset to run a test: 'MNIST', 'CIFAR'
    :param    dataType: specifies data type
    :param    train_data_path: specifies relative path to file with train data
    :param    test_data_path: specifies relative path to file with test data
    :param    dFormat: specifies source data format 'pkl' or 'csv'
    :param    train_size: specifies number of samples from dataset to train 
    :param    test_size: specifies number of samples from dataset to test
    :param    binarizationThreshold - a threshold for binarization of data
    model parameters
    :param    model: model of RBM to use: 'generative', 'joint' or 'discriminative'
    :param    batch size: size of minibatch 
    :param    lr: learning rate
    :param    lr_var: boolean value indicating if lr will be varied during learning
    :param    scal: sampling from Gaussian distribution with mean=0 and std var = scal
    :param    initBiasZero: if biases are initialized to zeros
    :param    nrHiddenUnits: number of hidden units
    :param    nrEpochs_p: number of training epochs
    :param    nrOfIter: number of iterations for sampling
    :param    randomState: random state used for random nr generating
    :param    errorThreshold: denotes stop condition for RBM training
    :param    momentum_p: momentum value
    :param    momentum_var: boolean value indicating if momentum will be varied during learning
    :param    CDk_p: number of Gibbs step used by contrastive divergence 
    :param    CDk_var: boolean value indicating if k for CD will be varied during learning
    :param    predictMethod: 1 - based on sampling target units, 2 - based on free energy computation
    visualizations
    :param    returnMSE: boolean value indicating if test run should return mse after training - used for plot function
    :param    plotMSE: boolean value indicating if MSE should be plotted
    :param    showFilt: boolean value indicating if filters should be displayed
    
    
        ...
"""       

def runTest(
            data = Dataset.MNIST.name,
            train_data_path = 'data/mnist.pkl.gz',
            test_data_path = 'data/mnist_test.csv',
            dataType = DataType.binary.name,
            dFormat = 'pkl',
            train_size = 100,
            test_size = 10,
            binarizationThreshold = 0.5,
            model = RBMType.joint.name,
            batch_size = 10,
            lr = 0.01,
            lr_var = False,
            scal = 0.01,
            initBiasZero=False,
            nrHiddenUnits_p = 700,
            nrEpochs_p = 100,
            nrOfIter = 1000,
            randomState = 1234,
            errorThreshold = 5,
            momentum_p = 0.0,
            momentum_var = False,
            CDk_p=1,
            CDk_var = False,
            predictMethod=2,
            returnMSE=False,
            plotMSE=False,
            showFilt=False
            ):
    
    #will modify the global variable
    global results
    # Generate random state to ensure deterministic, repeatable runs
    rnGen = np.random.RandomState(randomState)
    
    if (dataType == 'binary'):
        binary=True  
    else:
        binary=False
    # Read specified dataset from source files
    print('Reading data...')
    t0=time()
    # Case for MNIST data
    if (data == 'MNIST'):
        dataObj = MNIST();
        # Case of reading from cPickle format
        if dFormat == 'pkl':
            trainingSet, validationSet, testSet = dataObj.loadData(datafile = train_data_path)
            trainX, trainY = trainingSet
            validX, validY = validationSet
            #case of binary data
            if (dataType == 'binary'):
                trainX = binarize(trainX, threshold = binarizationThreshold)
                validX= binarize(validX, threshold = binarizationThreshold)
            if (train_size > 50000):
                trainX= np.concatenate((trainX, validX))
                trainY= np.concatenate((trainY, validY))
            
            trainX = trainX[:train_size]
            trainY = trainY[:train_size]
            # if using validation set as testing set
            #validX = validX[:test_size]
            #validY = validY[:test_size]
            
        else:
            #Case of reading from CSV
            if dFormat =='csv':
                dataSet = dataObj.readCSVDataFast(s=train_size, datafile = train_data_path)
                trainY = dataSet[:train_size,0]
                if (dataType == 'binary'):
                    trainX = binarize(scale(dataSet[:train_size,1:]), 
                                      threshold = binarizationThreshold)
                else:
                    trainX = scale(dataSet[:train_size,1:])
        
        #Test data are read from csv anyway
        testSet = dataObj.readCSVDataFast(datafile = test_data_path, s=test_size)
        testY = testSet[:test_size,0]
        if (dataType == 'binary'):
            testX = binarize(scale(testSet[:test_size,1:]), 
                                    threshold = binarizationThreshold)
        if (dataType == 'gaussian'):
            testX = scale(testSet[:test_size,1:])
    # Case for CIFAR data
    if (data == 'CIFAR'):
        data_dict1 = loadCIFAR('./data/cifar-10-batches-py/data_batch_1')
        data_dict2 = loadCIFAR('./data/cifar-10-batches-py/data_batch_1')
        data_dict3 = loadCIFAR('./data/cifar-10-batches-py/data_batch_1')
        data_dict4 = loadCIFAR('./data/cifar-10-batches-py/data_batch_1')
        data_dict5 = loadCIFAR('./data/cifar-10-batches-py/data_batch_1')
        examples = np.zeros((50000,3072))
        labels = np.zeros(50000)
        examples[:10000,:] = data_dict1["data"]
        examples[10000:20000,:] = data_dict2["data"]
        examples[20000:30000,:] = data_dict3["data"]
        examples[30000:40000,:] = data_dict4["data"]
        examples[40000:,:] = data_dict5["data"]
        #print examples
        labels[:10000] = data_dict1["labels"]
        labels[10000:20000] = data_dict2["labels"]
        labels[20000:30000] = data_dict3["labels"]
        labels[30000:40000] = data_dict4["labels"]
        labels[40000:] = data_dict5["labels"]
        read_time = time() - t0
        #print labels
        #data.plotCIFAR(examples[0,:])
        print "CIFAR data read"
    
    if(model == 'binomial'): 
        #initialize binomial RBM
        bRBM = rbmb.BinomialRestrictedBoltzmannMachine(3072,300,None)
        print('Training binomial model of RBM...')
        bRBM.train(examples,labels,labels[0],0.1,5)
        print "RBM trained"
        print('Sampling from binomial model of RBM...')
        data.plot(bRBM.sample(1000))
    #Initialize a chosen RBM model and perform train and sample operations
    if(model == 'generative'): 
        print('Training generative model of RBM...')
        nrVisibleUnits = len(trainX[1]);
        RBM1 = RBM(nrVisibleUnits, nrHiddenUnits_p, scal = scal, binary=binary, 
                   rnGen=rnGen)
        t1 = time() 
        RBM1.train(trainX, trainY, 5, lr, errorThreshold)
        train_time = time() - t1
        print("Read data time: %0.3fs" % read_time)
        print("Train time: %0.3fs" % train_time) 
        
        print('Perform sampling...')   
        t2=time()
        example = RBM1.sample(nrOfIter)
        sample_time = time() - t2
        print("Sampling time: %0.3fs" % sample_time) 
        dataObj.plot(example)
        #dataObj.visualize(RBM1.sample(nrOfIter))
    
    if(model == 'joint'):
        #transform training labels using LabelBinarization to binary vector
        trainY = dataObj.transformLabel(trainY)
        # set number of visible and target units
        nrVisibleUnits = len(trainX[1])
        nrTargetUnits = len(trainY[1])
        #split the training set into mini-batches of batch_size
        numOfBatches = trainX.shape[0] / batch_size
        #initialize jRBM
        jRBM1 = jRBM(nrVisibleUnits, nrHiddenUnits_p, nrTargetUnits, 
                     scal = scal, binary=binary, rnGen=rnGen, 
                     initBiasZero=initBiasZero)
        epoch = 0
        #initialize mean errors on data and labels
        mErrorX = 0
        mErrorY = 0
        mEs = np.zeros(nrEpochs_p+1)
        #for each epoch 
        print('Training joint-probabilities model of RBM...')
        t1 = time()
        while epoch < nrEpochs_p:
            t2 = time()
            epoch += 1
            #change some parameters while training progresses in varied models
            if lr_var:
                if epoch > 0.6 * nrEpochs_p:
                    lr = lr/10
                if epoch > 0.8 * nrEpochs_p:
                    lr = lr/100
            if momentum_var:
                if epoch > 0.7 * nrEpochs_p:
                    momentum_p = 0.5   
                if epoch > 0.9 * nrEpochs_p:
                    momentum_p = 0.9  
            if CDk_var:
                if epoch > 0.6 * nrEpochs_p:
                    CDk_p = 2   
                if epoch > 0.8 * nrEpochs_p:
                    CDk_p =3    
        #perform training based on CD for each minibatch
            for i in range(numOfBatches):
                #iterate to next batch
                #print i*batch_size
                batchX = trainX[i*batch_size:((i*batch_size)+batch_size)]
                batchY = trainY[i*batch_size:((i*batch_size)+batch_size)]
                #perform train on this batch and update weights globally
                gWVH, gWTH, gV, gT, gH, eX, eY = jRBM1.train(batchX, batchY, errorThreshold, k=CDk_p)
                
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
        if returnMSE:
            return mEs
        
        if plotMSE:
            #Plot mean squared error on data within epochs
            plt.figure()
            plt.title('Mean squared error for epochs')
            plt.plot(mEs, 'b', label='Model with varied momentum rate')
            plt.legend()
            plt.grid()
            plt.xlabel('Epoch')
            plt.ylabel('Mean-squared weight error')
            plt.xlim(xmin=1)
            plt.show()
        
        if showFilt:
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
            
        #Compute classification error   
        print('Performing classification on test data...')
        t3=time()
        if (predictMethod == 1):
            label = jRBM1.predict(testX,numOfIteration=nrOfIter)
        else:
            label = jRBM1.predict2(testX)
        predict_time = time()-t3
        #count how many had wrong predicted label
        #trainY = dataObj.inverseTransformLabel(trainY, set=True)
        #print trainY
        counter = 0
        for i in range(len(label)):
            #print "Reconstrlabel is %f, original label is%f" % (label[i],testY[i]) 
            if label[i] != testY[i]:
                test_data = testX[i]
                counter +=1
                #print wrongly predicted
                #print "Reconstrlabel is %f, original label is%f" % (label[i],testY[i]) 
                #save wrongly predicted
                scipy.misc.imsave('sample_pictures/wrong' + str(counter) +'_' 
                                  +str(testY[i]) + '_predicted_as_'+ 
                                  str(label[i]) +'.png', test_data.reshape(28,28))
        err = counter / float(len(label))
        acc = 1 - err
        print "Classification error is %0.3f" % err
        print "Accuracy is %0.3f" % acc
        print "Confusion matrix:" 
        print  confusion_matrix(label.astype(int),testY.astype(int))
        print "Classification report:" 
        print  classification_report(label.astype(int),testY.astype(int))
        print "Prediction time is %0.3fs" % predict_time
        # save results to plot later
        results.append((acc, train_time, predict_time))
        
    if(model == 'binomial'):
        bRBM = rbmb.BinomialRestrictedBoltzmannMachine(3072,300,None)
        bRBM.train(examples,labels,labels[0],0.1,5)
        print "RBM trained"
        data.plot(bRBM.sample(1000))
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
    print os.getcwd()
    #print os.path.relpath('')
    #print (RBMType.generative.name)
    
"""Function that plots and compares MSE for training 
with different parameters and different values """
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

"""Function that plots and compares MSE for training 
with a different values for a chosen parameter """   
def plotResults2(parameter = 'scal', 
                 val1 = 1,
                 val2 = 2,
                 val3 = 3,
                 val4 = 1,
                nr_epochs = 100):
    plt.figure()
    plt.title('Convergence comparison for different CD k-steps')
    mswe1 = runTest(CDk_p=val1, nrEpochs_p = nr_epochs, returnMSE = True)
    plt.plot(mswe1, 'b', label='%d' % val1)
    mswe2 = runTest(CDk_p=val2, nrEpochs_p = nr_epochs, returnMSE = True)
    plt.plot(mswe2, 'r', label='%d' % val2)
    mswe3 = runTest(CDk_p=val3, nrEpochs_p = nr_epochs, returnMSE = True)
    plt.plot(mswe3, 'g', label='%d' % val3)
    mswe4 = runTest(CDk_p=val4, CDk_var=True, nrEpochs_p = nr_epochs, returnMSE = True)
    plt.plot(mswe4, 'k--', label='varied')
    #mswe4 = runTest(momentum=val4, nrEpochs_p = nr_epochs)
    #plt.plot(mswe4, 'k', label='%d' % val4)

    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Mean-squared error on data')
    plt.xlim(xmin=1)
    plt.show()


"""Function that plots bar chart
and compares accuracy and times metrics for the chosen model """  
def compareMetrics():
    names=[]
    for val, name in (
        (1, "Sampling target units"),
        (2, "Computing free energy")):
        #(700, "Hidden units=700")):
        print('=' * 80)
        print(name)
        names.append((name))
        runTest(predictMethod=val, train_size = 5000,test_size = 1000)
    
    # make some plots
    global results
    indices = np.arange(len(results))
    #indices = 1
    
    results = [[x[i] for x in results] for i in range(3)]
    
    acc, train_time, predict_time = results
    train_time = np.array(train_time) / np.max(train_time)
    predict_time = np.array(predict_time) / np.max(predict_time)
    
    plt.figure(figsize=(12, 8))
    plt.title("Classification with different prediction methods")
    plt.barh(indices, acc, .2, label="accuracy", color='r')
    plt.barh(indices + .3, train_time, .2, label="train time", color='g')
    plt.barh(indices + .6, predict_time, .2, label="predict time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    
    for i, c in zip(indices, names):
        plt.text(-.3, i, c)
    
    plt.show()  

#runTest();
#simpleTest();
#miscTest();
#plotResults2()
compareMetrics();

