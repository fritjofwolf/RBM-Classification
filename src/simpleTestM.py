import rbm as rbm
import numpy as np
import data


mnist = data.MNIST()
trainingData = mnist.readCSVDataFast(s = 30000)
print "TrainingData loaded"
#print trainingData
RBM = [rbm.RestrictedBoltzmannMachine(784,1000,None) for i in range(10)]
#print trainingData[:,0]
#for i in range(10):
#	mnist.plot(data.binarize(trainingData[i,1:],128))
#print trainingData[trainingData[:,0] == trainedClass,0]
#print trainingData[trainingData[:,0] == trainedClass,0].shape
testData = trainingData[29000:,:]
trainingData2 = trainingData[:29000,:]
for i in range(1):
	trainedClass = 2
	specialTrainingData = trainingData2[trainingData2[:,0] == trainedClass,:]
	RBM[i].train(data.binarize(specialTrainingData[:,1:],128),specialTrainingData[:,0],trainedClass,0.005,2)
	print "Training von RBM " + str(i) + " beendet"

# Testing
#~ print "Start Testing"
#~ counter = 0
#~ for i in range(len(testData)):
	#~ if i % 100 == 0:
		#~ print i
	#~ a = np.array([RBM[j].compute_free_energy(testData[j,1:]) for j in range(10)])
	#~ if np.argmin(a) == testData[i,0]:
		#~ counter += 1
		#~ 
#~ print "Test finished"
#~ print "Accuracy is: " + str(counter/1000.)
RBM[0].sample(1000)
