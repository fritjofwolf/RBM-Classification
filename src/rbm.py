# Module with implementation of different models of Restricted Boltzmann Machines

import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from data import *


def sigmoid(x):
    if x < -100:
		return 0
    elif x > 100:
		return 1
    return 1 / (1 + math.exp(-x))

"""
Basic model of a Restricted Boltzmann Machine - a complete bipartite graph
Visible Units (V) represent observable data 
Hidden Units (H), in number of numOfHiddenUnits,
    capture the dependencies between the observed variable
In binary problem (binary=True) variables in V,H take values from {0;1}
Probability distribution under the model is given by Gibbs sampling
Wij (weight between i-th Visible Unit and j-th Hidden Unit) is real valued
    weight associated with edge between these units)
VisibleBias and HiddenBias are real valued bias terms associated 
    with i-th visible and j-th hidden units
    
Restricted Boltzmann Machine can be regarded as stochastic neural network,
where the nodes and edges correspond to neurons and synaptic commections, respectively.
The conditional probability of a single variable being one can be interpreted 
as the firing rate of a (stochastic) neuron with sigmoid activation function
"""
class RestrictedBoltzmannMachine(object):
    def __init__(self, numOfVisibleUnits, numOfHiddenUnits,  rnGen, 
                 weights = [], scal = 0.01, binary = True):
        #Parameters
        #bin:bool - if visible units are binary or normally distributed
        #scal: float - sample initial weights from Gaussian distribution (0,scal)
        
        #initialize random number generator
        if rnGen is None:
            # create a number generator
            rnGen = np.random.RandomState(1234)

        self.NumOfVisibleUnits = numOfVisibleUnits
        self.NumOfHiddenUnits = numOfHiddenUnits
        self.VisibleBiases = scal * np.random.randn(numOfVisibleUnits)
        self.HiddenBiases = scal * np.random.randn(numOfHiddenUnits)
        
        
        # Initialize weight matrix
        # Use small random values for the weights chosen from a zero-mean Gaussian with a standard deviation of scal.
        if weights != []:
            self.Weights = weights
        else:
            #self.Weights = np.random.random([numOfVisibleUnits, numOfHiddenUnits])
            self.Weights = scal * np.random.randn(numOfVisibleUnits, numOfHiddenUnits)
            
    # Train the RBM using the contrastive divergence algorithm
    # Input: trainingData - matrix of training examples, where each row represents an example
    def train(self, trainingData, label, classToTrain, learningRate, errorThreshold, 
              stopCondition = 'errorThreshold'):
        print("Start training")            
        counter = 0
        counterExamples = 0
        average_error_old = 1000000
        error = 10000
        #results = [[ for i in range(10)] for j in range(10)]
        # errorThreshold is termination condition
        while True:
                #print counter
                # train RBM only with examples from one class
                

                visible = np.transpose(trainingData[counter,:])
                hidden = np.zeros((self.NumOfHiddenUnits))
                visibleRecon = np.zeros((self.NumOfVisibleUnits))
                hiddenRecon = np.zeros((self.NumOfHiddenUnits))
                
                
                # Gibbs-Sampling
                #Sampling a new state h for the hidden neurons based on p(h|v)
                for i in range(self.NumOfHiddenUnits):
                    if np.random.random() < sigmoid(self.HiddenBiases[i] + np.inner(visible,self.Weights[:,i])):
                        hidden[i] = 1
                    else:
                        hidden[i] = 0
                #Sampling a new state v for the visible layer based on p(v|h)
                for i in range(self.NumOfVisibleUnits):
                    if np.random.random() < sigmoid(self.VisibleBiases[i] + np.inner(hidden,self.Weights[i,:])):
                        visibleRecon[i] = 1
                    else:
                        visibleRecon[i] = 0        
                for i in range(self.NumOfHiddenUnits):
                    if np.random.random() < sigmoid(self.HiddenBiases[i] + np.inner(visibleRecon,self.Weights[:,i])):
                        hiddenRecon[i] = 1
                    else:
                        hiddenRecon[i] = 0
                    
                # Update weights and biases
                self.Weights += learningRate * (np.outer(visible,hidden) - np.outer(visibleRecon,hiddenRecon))
                self.HiddenBiases += learningRate * (hidden - hiddenRecon)
                self.VisibleBiases += learningRate * (visible - visibleRecon)
                
                # Squared-error serves as indicator for the learning progress
                error += sum((visible-visibleRecon)**2)
                counter += 1
                counterExamples += 1
                
                if counter >= trainingData.shape[0]:
                    counter %= trainingData.shape[0]
                    print error / counterExamples
                    if abs(average_error_old - (error / counterExamples)) < errorThreshold:
                        break
                    average_error_old = error / counterExamples
                    counterExamples = 0
                    error = 0
                    
        print("End training")
    
    # Computes sample of the learned probability distribution
    def sample(self,numOfIteration):
        visible = np.random.randint(0,2,self.NumOfVisibleUnits)
        hidden = np.zeros((self.NumOfHiddenUnits))
        # Sample is computed by iteratively computing the activation of hidden and visible units
        for i in range(numOfIteration):
            for j in range(self.NumOfHiddenUnits):
                if np.random.random() < sigmoid(self.HiddenBiases[j] + np.inner(visible,self.Weights[:,j])):
                    hidden[j] = 1
                else:
                    hidden[j] = 0
                    
            for k in range(self.NumOfVisibleUnits):
                if np.random.random() < sigmoid(self.VisibleBiases[k] + np.inner(hidden,self.Weights[k,:])):
                    visible[k] = 1
                else:
                    visible[k] = 0
            visible[visible[:] == 0] = 255
            visible[visible[:] == 1] = 0
            
            if i % 1 == 0:
				scipy.misc.imsave('./sample_pictures/foo_' + str(i) + '.png', visible.reshape(28,28))
        
    # Computes the free energy of a given visible vector (formula due to Hinton "Practical Guide ...")      
    def compute_free_energy(self, visible):
        x = np.zeros(784)
        for j in range(self.NumOfHiddenUnits):
            x[j] = self.HiddenBiases[j] + np.inner(np.transpose(visible),self.Weights[:,j])
        return (-np.inner(visible,self.VisibleBiases) - sum([max(0,x[i]) for i in range(len(x))]))
        
    
"""
Version of Restricted Boltzmann Machine that models the joint distribution of 
inputs and target classes 
train a joint density model using a single RBM that has two sets of visible units
Overrides train method: uses contrastive divergence algorithm to calculate gradient
"""
class Joint(RestrictedBoltzmannMachine):
    def __init__(self, numOfVisibleUnits, numOfHiddenUnits,  numOfTargetUnits,
                 rnGen, weightsVH = [], weightsTH = [], 
                  scal = 0.01, binary = True, initBiasZero = False):
        RestrictedBoltzmannMachine.__init__(self, numOfVisibleUnits, 
                                            numOfHiddenUnits,
                                            rnGen = rnGen, 
                                            scal = scal, 
                                            binary=binary,
                                            )
        self.NumOfTargetUnits = numOfTargetUnits
        #Initialize weights
        # Use small random values for the weights chosen from a zero-mean Gaussian with a standard deviation of 0.01.
        
        if weightsVH != []: # weights between visible and hidden units
            self.WeightsVH = weightsVH
        else:
            #self.WeightsVH = np.random.random([numOfVisibleUnits, numOfHiddenUnits])
            self.WeightsVH = scal * np.random.randn(numOfVisibleUnits, numOfHiddenUnits)
        if weightsTH != []: # weights between target and hidden units
            self.WeightsTH = weightsTH
        else:
            #self.WeightsTH = np.random.random([numOfTargetUnits, numOfHiddenUnits])
            self.WeightsTH = scal * np.random.randn(numOfTargetUnits, numOfHiddenUnits)
        
        #Initialize weight, biases to zeros
        if initBiasZero:
            self.VisibleBiases = np.zeros(self.NumOfVisibleUnits, float)
            self.HiddenBiases = np.zeros(self.NumOfHiddenUnits, float)
            self.TargetBiases = np.zeros(self.NumOfHiddenUnits, float)
        else:
            #Initialize weight, biases to small numbers
            self.VisibleBiases = scal * np.random.randn(numOfVisibleUnits)
            self.HiddenBiases = scal * np.random.randn(numOfHiddenUnits)       
            self.TargetBiases = scal * np.random.randn(numOfTargetUnits)
     
    """
    Train the RBM using the contrastive divergence sampling
    Overrides train method of base class, 
    performs gradient estimation over mini-batch of samples
    Input: batch - a subset of training data, divided into X and Y (labels)
    k - number of iterations for CD algorthm
    Returns: Gradient approximation for weights, visible bias, hidden bias
    """
    def train(self, batchX, batchY, errorThreshold, k=1):
        gradientWVH = np.zeros(self.WeightsVH.shape, float)
        gradientWTH = np.zeros(self.WeightsTH.shape, float)
        gradientV= np.zeros(self.NumOfVisibleUnits, float)
        gradientH = np.zeros(self.NumOfHiddenUnits, float)
        gradientT = np.zeros(self.NumOfHiddenUnits, float)
        CDposVH = np.zeros(self.WeightsVH.shape, float)
        CDposTH = np.zeros(self.WeightsTH.shape, float)
        CDnegVH = np.zeros(self.WeightsVH.shape, float)
        CDnegTH = np.zeros(self.WeightsTH.shape, float)
        
        errorX = 0
        errorY = 0
        #iterate over samples in a batch
        for i in range(len(batchX)):
            #set state of visible units based on this data point
            visibleX = np.transpose(batchX[i,:])
            visibleY = np.transpose(batchY[i,:])
            visibleRecon = np.zeros((self.NumOfVisibleUnits))
            targetRecon = np.zeros((self.NumOfTargetUnits))
            #visibleRecon_prob = np.zeros((self.NumOfVisibleUnits))
            #targetRecon_prob = np.zeros((self.NumOfTargetUnits))
            hiddenRecon = np.zeros((self.NumOfHiddenUnits))
            hidden = np.zeros((self.NumOfHiddenUnits))
            #hidden_prob = np.zeros((self.NumOfHiddenUnits))

            #compute state for each hidden based on formula and visible
            for j in range(self.NumOfHiddenUnits):
                #Positive phase
                #Do sampling
                #hidden_prob[j] = sigmoid(self.HiddenBiases[j] + np.inner(visibleX,self.WeightsVH[:,j]) + np.inner(visibleY,self.WeightsTH[:,j]))
                if np.random.random() < sigmoid(self.HiddenBiases[j] + np.inner(visibleX,self.WeightsVH[:,j]) + np.inner(visibleY,self.WeightsTH[:,j])) :
                    hidden[j] = 1
                else:
                    hidden[j] = 0
            # for positive phase use correlations between data and hidden         
            CDposVH += np.outer(visibleX,hidden)
            CDposTH += np.outer(visibleY,hidden)
            
            #in k steps
            for step in range(k):
                if step == 0:
                    hid = hidden
                else:
                    hid = hiddenRecon
                #compute visible based on hidden units (reconstruction)
                for nv in range(self.NumOfVisibleUnits):
                    #visibleRecon_prob[nv] = sigmoid(self.VisibleBiases[nv] + np.inner(hid,self.WeightsVH[nv,:]))
                    if np.random.random() <  sigmoid(self.VisibleBiases[nv] + np.inner(hid,self.WeightsVH[nv,:])):
                        visibleRecon[nv] = 1
                    else:
                        visibleRecon[nv] = 0
                        
                #compute target based on hidden units (reconstruction)
                for nt in range(self.NumOfTargetUnits):
                    #targetRecon[nt] = sigmoid(self.TargetBiases[nt] + np.inner(hid,self.WeightsTH[nt,:]))
                    if np.random.random() < sigmoid(self.TargetBiases[nt] + np.inner(hid,self.WeightsTH[nt,:])):
                        targetRecon[nt] = 1
                    else:
                        targetRecon[nt] = 0
                    
                #compute hidden states again 
                for j in range(self.NumOfHiddenUnits):
                    #For the last update of hidden units use the probability instead of stochastic binary states 
                    if step == k-1:
                        hiddenRecon[j] = sigmoid(self.HiddenBiases[j] + np.inner(visibleRecon,self.WeightsVH[:,j]) + np.inner(targetRecon,self.WeightsTH[:,j]))
                    else:
                        #Do sampling
                        if np.random.random() < sigmoid(self.HiddenBiases[j] + np.inner(visibleRecon,self.WeightsVH[:,j]) + np.inner(targetRecon,self.WeightsTH[:,j])):
                            hiddenRecon[j] = 1
                        else:
                            hiddenRecon[j] = 0
            
            CDnegVH += np.outer(visibleRecon,hiddenRecon)
            CDnegTH += np.outer(targetRecon,hiddenRecon)
            
            # Squared-error serves as indicator for the learning progress
            errorX += sum((visibleX-visibleRecon)**2)
            errorY += sum((visibleY-targetRecon)**2)

        #compute average for batch
        CDposVH /= len(batchX)
        CDposTH /= len(batchX)
    
        CDnegVH /= len(batchX)
        CDnegTH /= len(batchX)
        
        #compute mean error for the batch
        errorX /= len(batchX)
        errorY /= len(batchX)
    
        #compute gradients for this batch
        gradientWVH = CDposVH - CDnegVH
        gradientWTH = CDposTH - CDnegTH  
        gradientV = (visibleX - visibleRecon).mean(axis=0)
        gradientT = (visibleY - targetRecon).mean(axis=0)
        gradientH = (hidden - hiddenRecon).mean(axis=0)   
        
        #print errorX, error
	#dataObj = MNIST()
	#dataObj.plot(visibleX)
        return gradientWVH, gradientWTH, gradientV, gradientT, gradientH, errorX, errorY
    
    """   
   Updates weights based on gradients and learning rate
   weightDecay - 'l2' or 'l1' method of weight penalization
    """
    def updateWeight(self, lR, gradientWVH, gradientWTH, gradientV, gradientH,
                     gradientT, weightDecay = 'l2', momentum=1.0, l2= 0.0): 
        
        gradientWVH *= 1 -momentum
        gradientWTH *= 1 -momentum
        gradientH *= 1 -momentum
        gradientV *= 1 -momentum
        gradientT *= 1 -momentum  
        
        gradientWVH += momentum * (gradientWVH - l2*self.WeightsVH)
        gradientWTH += momentum * (gradientWTH - l2*self.WeightsTH)
        gradientH += momentum * (gradientH - l2*self.HiddenBiases)
        gradientV += momentum * (gradientV - l2*self.VisibleBiases)
        gradientT += momentum  * (gradientT - l2*self.TargetBiases) 
        
        self.WeightsVH += lR * gradientWVH
        self.WeightsTH += lR * gradientWTH
        self.HiddenBiases += lR * gradientH
        self.VisibleBiases += lR * gradientV
        self.TargetBiases += lR * gradientT
    
    """
    Computes sample of a label given data (image)
    The label corresponding to an input image is obtained by fixing the visible variables
    corresponding to the image and then sampling the remaining visible variables corresponding to the labels from
    the joined probability distribution of images and labels modeled by the RBM
    """
    def sample(self,testSampleX, numOfIteration):

        visibleX = testSampleX
        visibleY = np.zeros((self.NumOfTargetUnits))
        hidden = np.zeros((self.NumOfHiddenUnits))
        # Sample is computed by iteratively computing the activation of hidden and visible units
        for j in range(numOfIteration):
            for i in range(self.NumOfHiddenUnits):
                if np.random.random() < sigmoid(self.HiddenBiases[i] + np.inner(visibleX,self.WeightsVH[:,i])+ np.inner(visibleY,self.WeightsTH[:,i])):
                    hidden[i] = 1
                else:
                    hidden[i] = 0
            for i in range(self.NumOfTargetUnits):
                if j == numOfIteration-1:
                    visibleY[i] = sigmoid(self.TargetBiases[i] + np.inner(hidden, self.WeightsTH[i,:]))
                else:
                    if np.random.random() < sigmoid(self.TargetBiases[i] + np.inner(hidden, self.WeightsTH[i,:])):
                        visibleY[i] = 1
                    else:
                        visibleY[i] = 0  
        return visibleY  
    
    """
    Performs classification on given dataset
    uses sample method
    return predicted labels
    """
    def predict(self,testsetX, numOfIteration):
        labels = np.zeros(len(testsetX))
        #print testsetX.shape
        for i in range(len(testsetX)):
            reconY = self.sample(testsetX[i],numOfIteration = numOfIteration)
            #for j in range(len(reconY)):
                #if reconY[j] == 1:
                    #labels[i] = j  
                    #print labels[i] 
            #returns label that has the highest prob
            labels[i] = reconY.argmax(axis=0) 
        return labels
    
    # Computes the free energy of a given visible vector (formula due to Hinton "Practical Guide ...") 
    # Overrides method in the main class     
    def compute_free_energy(self, visibleX, visibleY):
        x = np.zeros(len(visibleX))
        for j in range(self.NumOfHiddenUnits):
            x[j] = self.HiddenBiases[j] + np.inner(np.transpose(visibleX),self.WeightsVH[:,j])+ np.inner(np.transpose(visibleY),self.WeightsTH[:,j])
        return (-np.inner(visibleX,self.VisibleBiases) - sum([max(0,x[i]) for i in range(len(x))]))
    
    """
    Performs classification on given dataset
    uses free energy method (Hinton, p.17)
    return predicted labels
    """
    def predict2(self,testsetX):
        labels = np.zeros(len(testsetX))
        #print testsetX.shape
        for i in range(len(testsetX)):
            min_fe = 99999
            label_min_fe=None
            visibleX = testsetX[i]
            #for each label
            for j in range(self.NumOfTargetUnits):
                visibleY = np.zeros(self.NumOfTargetUnits)
                visibleY[j] = 1
                #print visibleY
                #compute free energy
                fe = self.compute_free_energy(visibleX, visibleY)
                if fe < min_fe:
                    min_fe = fe
                    label_min_fe = j
            #returns label with minimal free energy
            labels[i] = label_min_fe
        return labels

"""
Version of Restricted Boltmann Machine that:
-is a modified version of Joint RBM version
-uses joint probabilities to train
-uses contrastive divergence in sampling
Modifications
-optimizes directly p(y|x) instead of p(y,x)
-gradient is computed exactly, not estimated
-implements predictClass method
"""
class Discriminative(Joint):
    
    """
    Overrides train method in Joint
    computes gradient exactly, not by estimation
    then used in stochastic gradient descent optimization
    """
    def train(self, visibleX, visibleY, learningRate, errorThreshold, k=1, 
        weightDecay='l2', momentum=0.5, stopCondition='epochNumber', 
        nrEpochs=10000):
        pass
            
    """
    Predicts class number based on input data
    After training, each possible label is tried in turn with a test vector 
    and the one that gives lowest free energy is chosen as the most likely class
    """
    def predictClass(self, inputData):
        #TODO
        label=0
        return label
"""    
Model of a RBM whose visible units are binomial units, i.e. they can 
model an integer between 0 and N
"""
class BinomialRestrictedBoltzmannMachine(object):
    def __init__(self, numOfVisibleUnits, numOfHiddenUnits,  rnGen, 
                 weights = [], scal = 0.01):
        #Parameters
        #bin:bool - if visible units are binary or normally distributed
        #scal: float - sample initial weights from Gaussian distribution (0,scal)
        
        #initialize random number generator
        if rnGen is None:
            # create a number generator
            rnGen = np.random.RandomState(1234)

        self.NumOfVisibleUnits = numOfVisibleUnits
        self.NumOfHiddenUnits = numOfHiddenUnits
        self.VisibleBiases = scal * np.random.randn(numOfVisibleUnits)
        self.HiddenBiases = scal * np.random.randn(numOfHiddenUnits)
        
        
        # Initialize weight matrix
        # Use small random values for the weights chosen from a zero-mean Gaussian with a standard deviation of scal.
        if weights != []:
            self.Weights = weights
        else:
            #self.Weights = np.random.random([numOfVisibleUnits, numOfHiddenUnits])
            self.Weights = scal * np.random.randn(numOfVisibleUnits, numOfHiddenUnits)
            
    # Train the RBM using the contrastive divergence algorithm
    # Input: trainingData - matrix of training examples, where each row represents an example
    def train(self, trainingData, label, classToTrain, learningRate, errorThreshold, 
              stopCondition = 'errorThreshold'):
        print("Start training")            
        counter = 0
        error = 10000
        #results = [[ for i in range(10)] for j in range(10)]
        # errorThreshold is termination condition
        while error > errorThreshold:
            # train RBM only with examples from one class
                if label[counter] != classToTrain:
                    counter += 1
                    counter %= trainingData.shape[0]
                    continue

                visible = np.transpose(trainingData[counter,:])
                hidden = np.zeros((self.NumOfHiddenUnits))
                visibleRecon = np.zeros((self.NumOfVisibleUnits))
                hiddenRecon = np.zeros((self.NumOfHiddenUnits))
                
                # Gibbs-Sampling
                # Sampling a new state h for the hidden neurons based on p(h|v)
                for i in range(self.NumOfHiddenUnits):
                    if np.random.random() < sigmoid(self.HiddenBiases[i] + np.inner(visible,self.Weights[:,i])):
                        hidden[i] = 1
                    else:
                        hidden[i] = 0
                        
                # Sampling a new state v for the visible layer based on p(v|h)
                # Since the visible units are binomial to model an int between 
                # 0 and 255. The probability p is computed only once and then
                # the value is computed by simultating 256 units 
                for i in range(self.NumOfVisibleUnits):
                    p = sigmoid(self.VisibleBiases[i] + np.inner(hidden,self.Weights[i,:]))
                    #print p
                    A = np.random.rand(256)
                    visibleRecon[i] = sum(A <p)
                    #print(visibleRecon[i],p)
                for i in range(self.NumOfHiddenUnits):
                    if np.random.random() < sigmoid(self.HiddenBiases[i] + np.inner(visibleRecon,self.Weights[:,i])):
                        hiddenRecon[i] = 1
                    else:
                        hiddenRecon[i] = 0
                    
                # Update weights and biases
                self.Weights += learningRate * (np.outer(visible,hidden) - np.outer(visibleRecon,hiddenRecon))
                self.HiddenBiases += learningRate * (hidden - hiddenRecon)
                self.VisibleBiases += learningRate * (visible - visibleRecon)
                
                # Squared-error serves as indicator for the learning progress
                error = sum(abs(visible-visibleRecon))
                #print("Hidden biases ", self.HiddenBiases)
                #print("Visible biases ", self.VisibleBiases)
                #print("Weights are ", self.Weights)
                print("Error is ", error)
                counter += 1
                counter %= trainingData.shape[0]
        print("End training")
    
    # Computes sample of the learned probability distribution
    def sample(self,numOfIteration):
        visible = np.random.randint(0,2,self.NumOfVisibleUnits)
        hidden = np.zeros((self.NumOfHiddenUnits))
        # Sample is computed by iteratively computing the activation of hidden and visible units
        for i in range(numOfIteration):
            for i in range(self.NumOfHiddenUnits):
                if np.random.random() < sigmoid(self.HiddenBiases[i] + np.inner(visible,self.Weights[:,i])):
                    hidden[i] = 1
                else:
                    hidden[i] = 0
                    
            for i in range(self.NumOfVisibleUnits):
                if np.random.random() < sigmoid(self.VisibleBiases[i] + np.inner(hidden,self.Weights[i,:])):
                    visible[i] = 1
                else:
                    visible[i] = 0
                    
        return visible

