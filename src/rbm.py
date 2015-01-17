# Module with Restricted Boltzmann Machines implementations
import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Basic model of a Restricted Boltzmann Machine (as a complete bipartite graph)
class RestrictedBoltzmannMachine(object):
    def __init__(self, numOfVisibleUnits, numOfHiddenUnits, weights = [], scal = 0.01, binary = True):
        #Parameters
        #bin:bool - if visible units are binary or normally distributed
        #scal: float - sample initial weights from Gaussian distribution (0,scal)
        
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
    def train(self, trainingData, label, classToTrain, learningRate, errorThreshold, momentum = 0):
        print("Start training")            
        counter = 0
        error = 10000
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
                for i in range(self.NumOfHiddenUnits):
                    if np.random.random() < sigmoid(self.HiddenBiases[i] + np.inner(visible,self.Weights[:,i])):
                        hidden[i] = 1
                    else:
                        hidden[i] = 0
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
                error = sum((visible-visibleRecon)**2)
                if counter % 10 == 0:
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
    
# Restricted Boltzmann Machine modeling the joint distribution of inputs and target classes 
# with hidden units, input and target classes  
class Discriminative(RestrictedBoltzmannMachine):
    def __init__(self, numOfVisibleUnits, numOfHiddenUnits, numOfTargetUnits, weightsVH = [], weightsTH = [], scal = 0.01, binary = True):
        RestrictedBoltzmannMachine.__init__(self, numOfVisibleUnits, numOfHiddenUnits, scal, bin)
        #self.NumOfVisibleUnits = numOfVisibleUnits
        #self.NumOfHiddenUnits = numOfHiddenUnits
        self.NumOfTargetUnits = numOfTargetUnits
        #self.VisibleBiases = np.random.random(numOfVisibleUnits)
        #self.HiddenBiases = np.random.random(numOfHiddenUnits)       
        self.TargetBiases = np.random.random(numOfTargetUnits)
        
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
            self.WeightsVH = scal * np.random.randn(numOfVisibleUnits, numOfHiddenUnits)
    
    # TODO: ANpassen        
    # Train the RBM using the contrastive divergence algorithm generalized to input and target
    # Input: trainingData - matrix of training examples, where each row represents an example
    def train(self, trainingData, learningRate, errorThreshold):
        counter = 0
        error = 10000
        while error > errorThreshold:
            visible = trainingData[counter]
            #Set the hidden biases to 0
            hidden = np.zeros((self.NumOfHiddenUnits,1))
            visibleRecon = np.zeros((self.NumOfVisibleUnits,1))
            hiddenRecon = np.zeros((self.NumOfHiddenUnits,1))
                
            # Gibbs-Sampling
            for i in range(self.NumOfHiddenUnits):
                if np.random.random() < sigmoid(self.HiddenBiases[i] + sum(visible*self.Weights[i,:])):
                    hidden[i] = 1
                else:
                    hidden[i] = 0
            for i in range(self.NumOfVisibleUnits):
                if np.random.random() < sigmoid(self.VisibleBiases[i] + sum(hidden*self.Weights[i,:])):
                    visibleRecon[i] = 1
                else:
                    visibleRecon[i] = 0        
                for i in range(self.NumOfHiddenUnits):
                    if np.random.random() < sigmoid(self.HiddenBiases[i] + sum(visibleRecon*self.Weights[i,:])):
                        hiddenRecon[i] = 1
                    else:
                        hiddenRecon[i] = 0
                    
                # Update weights and biases
                self.Weights += learningRate * (np.outer(visible,hidden) - np.outer(visibleRecon,hiddenRecon))
                self.HiddenBiases += learningRate * (hidden - hiddenRecon)
                self.VisibleBiases += learningRate * (visible - visibleRecon)
                
                # Squared-error serves as indicator for the learning progress
                error = sum((visible-visibleRecon)**2)
            print error
            counter += 1
            counter %= trainingData.shape[0]
    
    
    # Computes sample of the learned probability distribution
    def sample(self,numOfIteration):
        visible = np.random.randint(0,2,self.NumOfVisibleUnits)
        hidden = np.zeros((self.NumOfHiddenUnits,1))
        # Sample is computed by iteratively computing the activation of hidden and visible units
        for i in range(numOfIteration):
            for i in range(self.NumOfHiddenUnits):
                if np.random.random() < sigmoid(self.HiddenBiases[i] + sum(visible*self.Weights[i,:])):
                    hidden[i] = 1
                else:
                    hidden[i] = 0
                    
            for i in range(self.NumOfVisibleUnits):
                if np.random.random() < sigmoid(self.VisibleBiases[i] + sum(hidden*self.Weights[i,:])):
                    visible[i] = 1
                else:
                    visible[i] = 0
                    
        return visible
