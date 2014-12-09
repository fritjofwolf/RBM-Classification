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
	# 		 label - vector of labels belonging to the training examples
	def train(self, trainingData, label):
		pass
		
	
	# Computes sample of the learned probability distribution
	def sample(self):
		visible = np.random.randint(0,2,self.NumOfVisibleUnits)
		hidden = np.zeros((self.NumOfHiddenUnits,1))
		for i in range(self.NumOfHiddenUnits):
			if np.random.random() < sigmoid(self.HiddenBiases[i] + sum(visible*self.Weights[i,:])):
				hidden[i] = 1
			else:
				hidden[i] = 0
		for  i in range(100):
			
		return visible
		
