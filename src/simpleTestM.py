import rbm as rbm
import numpy as np
import data


mnist = data.MNIST()
trainingData = mnist.readCSVDataFast()
print "TrainingData loaded"
#print trainingData
RBM = [rbm.RestrictedBoltzmannMachine(784,100,None) for i in range(10)]
#print trainingData[:,0]
#for i in range(10):
#	mnist.plot(data.binarize(trainingData[i,1:],128))
trainedClass = 4
#print trainingData[trainingData[:,0] == trainedClass,0]
#print trainingData[trainingData[:,0] == trainedClass,0].shape
testData = trainingData[9000:,:]
trainingData2 = trainingData[:9000,:]
for i in range(10):
	trainedClass = i
	specialTrainingData = trainingData2[trainingData2[:,0] == trainedClass,:]
	RBM[i].train(data.binarize(specialTrainingData[:,1:],128),specialTrainingData[:,0],trainedClass,0.05,50)
	print "Training von RBM " + str(i) + " beendet"

# Testing
counter = 0
for i in range(len(testData)):
	a = np.array([RBM[i].compute_free_energy(testData[i,1:])])
	if np.argmin(a) == testData[i,0]:
		counter += 1
		
print "Test finished"
print "Accuracy is:" + str(counter/1000.)
#RBM.sample(100)
