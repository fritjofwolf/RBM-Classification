# Module with Restricted Boltzmann Machines implementations

import math
import numpy as np

def sigmoid(x):
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
    def __init__(self, numOfVisibleUnits, numOfHiddenUnits,  rnGen, weights = [], scal = 0.01, binary = True):
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
    def train(self, trainingData, label, classToTrain, learningRate, errorThreshold):
        print("Start training")            
        counter = 0
        error = 10000
        #results = [[ for i in range(10)] for j in range(10)]
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
    def __init__(self, numOfVisibleUnits, numOfHiddenUnits, numOfTargetUnits, 
                 weightsVH = [], weightsTH = [], scal = 0.01, binary = True, momentum = 0):
        RestrictedBoltzmannMachine.__init__(self, numOfVisibleUnits, 
                                            numOfHiddenUnits, scal = scal, binary=binary)
        #self.NumOfVisibleUnits = numOfVisibleUnits
        #self.NumOfHiddenUnits = numOfHiddenUnits
        self.NumOfTargetUnits = numOfTargetUnits
        self.momentum = momentum
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
        
        gradientW, gradientV, gradientH = self.contrastiveDivergence(trainingData)
        
        # TO DO
        #Update Visible Bias
        #self.VisibleBiases += 
        #Update Hidden Bias
        
        #Update Weights
        
        """
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
    """
    
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
    """
    K-step Contrastive divergence 
    Input: training batch of Visible and Hidden Units
    Returns: Gradient approximation for weights, visible bias, hidden bias
    """ 
    def contrastiveDivergence(self, batch):
       
        #Initialize weight, biases to zeros
        self.VisibleBiases = np.zeros(self.NumOfVisibleUnits, float)
        self.HiddenBiases = np.zeros(self.NumOfHiddenUnits, float)
        self.Weights = np.zeros(self.Weights.shape, float)
        
        #for all the visible units in batch
        #in k steps
        #TO DO
        
        
        gradientW = 0
        gradientV = 0
        gradientH = 0
        return gradientW, gradientV, gradientH   
