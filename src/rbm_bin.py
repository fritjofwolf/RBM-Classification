# Class definition of an RBM whose visible units are binomial (with N = 256)

import numpy as np
import math

def sigmoid(x):
	if x < -100:
		return 0
	if x > 100:
		return 1
	return 1 / (1 + math.exp(-x))
    
# Model of a RBM whose visible units are binomial units, i.e. they can 
# model an integer between 0 and N
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
                
		# Variable learning rate should ensure better convergence
		learningRate *= 0.95

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
