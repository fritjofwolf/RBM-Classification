import rbm_bin as rbmb
import numpy as np
import data

bRBM = rbmb.BinomialRestrictedBoltzmannMachine(3072,300,None)
data_dict1 = data.loadCIFAR('./data/cifar-10-batches-py/data_batch_1')
data_dict2 = data.loadCIFAR('./data/cifar-10-batches-py/data_batch_1')
data_dict3 = data.loadCIFAR('./data/cifar-10-batches-py/data_batch_1')
data_dict4 = data.loadCIFAR('./data/cifar-10-batches-py/data_batch_1')
data_dict5 = data.loadCIFAR('./data/cifar-10-batches-py/data_batch_1')
examples = np.zeros((50000,3072))
labels = np.zeros(50000)
examples[:10000,:] = data_dict1["data"]
examples[10000:20000,:] = data_dict2["data"]
examples[20000:30000,:] = data_dict3["data"]
examples[30000:40000,:] = data_dict4["data"]
examples[40000:,:] = data_dict5["data"]
print examples
labels[:10000] = data_dict1["labels"]
labels[10000:20000] = data_dict2["labels"]
labels[20000:30000] = data_dict3["labels"]
labels[30000:40000] = data_dict4["labels"]
labels[40000:] = data_dict5["labels"]
print labels
#data.plotCIFAR(examples[0,:])
print "Data read"
bRBM.train(examples,labels,labels[0],0.1,5)
print "RBM trained"
data.plot(bRBM.sample(1000))
