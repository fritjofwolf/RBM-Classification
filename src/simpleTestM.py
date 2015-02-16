import rbm as rbm
import numpy as np
import data


mnist = data.MNIST()
trainingData = mnist.readCSVDataFast()
print "TrainingData loaded"
#print trainingData
RBM = rbm.RestrictedBoltzmannMachine(784,100,None)
#print trainingData[:,0]
#for i in range(10):
#	mnist.plot(data.binarize(trainingData[i,1:],128))
trainedClass = 4
#print trainingData[trainingData[:,0] == trainedClass,0]
#print trainingData[trainingData[:,0] == trainedClass,0].shape
specialTrainingData = trainingData[trainingData[:,0] == trainedClass,:]
RBM.train(data.binarize(specialTrainingData[:,1:],128),specialTrainingData[:,0],trainedClass,0.05,1)
print "Training beendet"

RBM.sample(100)
