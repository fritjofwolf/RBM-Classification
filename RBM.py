# Model of a Restricted Boltzmann Machine (as a complete bipartite graph)
import math
import numpy as np

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class RestrictedBoltzmannMachine(object):
	def __init__(self, numOfVisibleUnits, numOfHiddenUnits, weights = []):
		self.NumOfVisibleUnits = numOfVisibleUnits
		self.NumOfHiddenUnits = numOfHiddenUnits
		self.VisibleBiases = np.random.random(numOfVisibleUnits)
		self.HiddenBiases = np.random.random(numOfHiddenUnits)
		if weights != []:
			self.Weights = weights
		else:
			self.Weights = np.random.random([numOfVisibleUnits, numOfHiddenUnits])
			
	# Train the RBM using the contrastive divergence algorithm
	# Input: trainingData - matrix of training examples, where each row represents an example
	# 		label - vector of labels belonging to the training examples
	def train(self, trainingData, label, learningRate):
		pass
		
	
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
		
