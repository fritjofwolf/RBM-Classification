from RBM import RestrictedBoltzmannMachine as RBM
from data import *
import math
import numpy as np

#RBM1 = RBM(5,2)
#print RBM1.Weights

#Test on loading, binarizing/scaling and printing/saving MNIST train data
printData(binarizeH(readData("mnist_train.csv", n=100)));
#saveData("out.csv", binarizeH(readData("mnist_train.csv", n=100)));