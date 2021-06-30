"""
Created on Sat May 22 15:22:14 2021

Author: Juan Rios

This file is the main entry point for the program. This file loads the training
and validation data, initiates the RBM, trains the RBM, and classifies the 
validation data.
"""

"""
Known Issues:
    2. why does gibbs sampling take so long
    3. Why does validating the data take so long
"""

#%%

from numpy import loadtxt
import pandas as pd
from dataLoader import dataLoader as dl
from RBM import RBM
import math
import time


#%%
# 1. Declaring RBM parameters

n = 10 # the number of hidden units
m = 27613 # the max number of visible units
k = 2 # number of possible values for V
T = 1 # number of gibbs sampling cycles

batchSize = 1000 # number of samples per batch
epochs = 50 # how many epochs to train for
alpha = 1/batchSize # learning rate alpha, default 0.01/batchSize

#%%
# 2. Loading the training and validation data
#tic = time.perf_counter()
#toc = time.perf_counter()
#print(f"loaded val data in {toc - tic:0.4f} seconds")

trainData = dl.loadData("trainData.txt")
trainMap = dl.loadData("trainMap.txt")
#valData = dl.loadData("valData.txt")

numBatches =  math.ceil(len(trainData)/batchSize)

#%%

# 3. initialize and train the model

model = RBM(k,m,n, alpha)
#model.validate(valData, trainData, trainMap)
model.train(epochs, numBatches, batchSize,  trainData, trainMap, T)


