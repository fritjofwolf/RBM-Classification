import rbm as rbm
import numpy as np
import data


mnist = data.MNIST()
trainingData = mnist.readCSVDataFast()
print "TrainingData loaded"
#print trainingData
RBM = rbm.RestrictedBoltzmannMachine(784,300,None)
#print trainingData[:,0]
RBM.train(trainingData[:,1:],trainingData[:,0],4,0.05,5)
RBM.sample()
