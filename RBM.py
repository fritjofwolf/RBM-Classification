# Model of a Restricted Boltzmann Machine (as a complete bipartite graph)
import math
import numpy as np

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class RestrictedBoltzmannMachine(object):
	def __init__(self, numOfVisibleUnits, numOfHiddenUnits, weights = [], scal = 0.01, bin = True):
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
		
