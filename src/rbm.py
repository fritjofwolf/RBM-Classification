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
    
"""
Version of Restricted Boltzmann Machine that models the joint distribution of 
inputs and target classes 
train a joint density model using a single RBM that has two sets of visible units
Overrides train method: uses contrastive divergence algorithm to calculate gradient
"""
class Joint(RestrictedBoltzmannMachine):
    def __init__(self, numOfVisibleUnits, numOfHiddenUnits,  numOfTargetUnits,
                 rnGen, weightsVH = [], weightsTH = [], 
                  scal = 0.01, binary = True):
        RestrictedBoltzmannMachine.__init__(self, numOfVisibleUnits, 
                                            numOfHiddenUnits,
                                            rnGen - rnGen, 
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
        #self.VisibleBiases = np.zeros(self.NumOfVisibleUnits, float)
        #self.HiddenBiases = np.zeros(self.NumOfHiddenUnits, float)
        #self.TargetBiases = np.zeros(self.NumOfHiddenUnits, float)
        #self.WeightsTH = np.zeros(self.WeightsTH.shape, float)
        #self.WeightsVH = np.zeros(self.WeightsVH.shape, float)

        #self.NumOfVisibleUnits = numOfVisibleUnits
        #self.NumOfHiddenUnits = numOfHiddenUnits
        
        #Initialize weight, biases to small numbers
        self.VisibleBiases = scal * np.random.randn(numOfVisibleUnits)
        self.HiddenBiases = scal * np.random.randn(numOfHiddenUnits)       
        self.TargetBiases = scal * np.random.randn(numOfTargetUnits)
        
    # TODO: ANpassen        
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
        
        #iterate over samples in a batch
        for i in range(len(batchX)):
            #set state of visible units based on this data point
            visibleX = np.transpose(batchX[i,:])
            visibleY = np.transpose(batchY[i,:])
            visibleRecon = np.zeros((self.NumOfVisibleUnits))
            targetRecon = np.zeros((self.NumOfTargetUnits))
            hiddenRecon = np.zeros((self.NumOfHiddenUnits))
            hidden = np.zeros((self.NumOfHiddenUnits))

            #compute state for each hidden based on formula and visible
            for j in range(self.NumOfHiddenUnits):
                #Positive phase
                #Do sampling
                if np.random.random() < sigmoid(self.HiddenBiases[j] + np.inner(visibleX,self.WeightsVH[:,j]) + np.inner(visibleY,self.WeightsTH[:,j])):
                    hidden[j] = 1
                else:
                    hidden[j] = 0
                    
            CDposVH += np.outer(visibleX,hidden)
            CDposTH += np.outer(visibleY,hidden)
            
            #in k steps
            for step in k:
                #compute visible based on hidden units (reconstruction)
                for nv in range(self.NumOfVisibleUnits):
                    if np.random.random() < sigmoid(self.VisibleBiases[nv] + np.inner(hidden,self.WeightsVH[nv,:])):
                        visibleRecon[nv] = 1
                    else:
                        visibleRecon[nv] = 0
                        
                #compute target based on hidden units (reconstruction)
                for nt in range(self.NumOfTargetUnits):
                    if np.random.random() < sigmoid(self.TargetBiases[nt] + np.inner(hidden,self.WeightsTH[nt,:])):
                        targetRecon[nt] = 1
                    else:
                        targetRecon[nt] = 0
                    
                #compute hidden states again 
                for j in range(self.NumOfHiddenUnits):
                    #Use weights between hidden and target
                    #if z < self.NumOfTargetUnits:
                        #Do sampling
                        if np.random.random() < sigmoid(self.HiddenBiases[j] + np.inner(visibleX,self.WeightsVH[:,j]) + np.inner(visibleY,self.WeightsTH[:,j])):
                            hiddenRecon[j] = 1
                        else:
                            hiddenRecon[j] = 0
                
            
            CDnegVH += np.outer(visibleRecon,hiddenRecon)
            CDnegTH += np.outer(targetRecon,hiddenRecon)

        #compute average for batch
        CDposVH /= len(batchX)
        CDposTH /= len(batchX)
    
        CDnegVH /= len(batchX)
        CDnegTH /= len(batchX)
    
        #compute gradients for this batch
        gradientWVH = CDposVH - CDnegVH
        gradientWTH = CDposTH - CDnegTH  
        gradientV = visibleX - visibleRecon
        gradientT = visibleY - targetRecon
        gradientH = hidden - hiddenRecon      
        
        # Squared-error serves as indicator for the learning progress
        errorX = sum((visibleX-visibleRecon)**2)
        errorY = sum((visibleY-targetRecon)**2)
        print errorX, errorY
    
        return gradientWVH, gradientWTH, gradientV, gradientT, gradientH, errorX, errorY
    
    """   
   Updates weights based on gradients and learning rate
   weightDecay - 'l2' or 'l1' method of weight penalization
    """
    def updateWeight(self, lR, gradientWVH, gradientWTH, gradientV, gradientH,
                     gradientT, weightDecay = 'l2', momentum=0.5): 
        
        self.WeightsVH += lR * gradientWVH
        self.WeightsTH += lR * gradientWTH
        self.HiddenBiases += lR * gradientH
        self.VisibleBiases += lR * gradientV
        self.TargetBiases += lR * gradientT
                    
                   
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
