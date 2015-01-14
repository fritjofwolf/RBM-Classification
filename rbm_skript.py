from RBM import RestrictedBoltzmannMachine as RBM
from data import *
import math
import numpy as np

#RBM1 = RBM(5,2)
#print RBM1.Weights

#Test on loading, binarizing/scaling and printing/saving MNIST train data
#printData(binarizeH(readData("mnist_train.csv", n=100)));
dataSet = readDataFast("mnist_train.csv")
label = dataSet[:,0]
trainingData = binarizeH(dataSet[:,1:])

RBM = RBM(784,100)
RBM.train(trainingData, label, 2, 0.05,5)
visualizeMNIST(RBM.sample(100))
#visualizeMNIST(trainingData[0,:])
#saveData("out.csv", binarizeH(readData("mnist_train.csv", n=100)));
