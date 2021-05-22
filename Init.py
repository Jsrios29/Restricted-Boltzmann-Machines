"""
Created on Sat May 22 15:22:14 2021

Author: Juan Rios

This file is the main entry point for the program. This file loads the training
and validation data, initiates the RBM, trains the RBM, and classifies the 
validation data.
"""

#%%

from numpy import loadtxt
import pandas as pd
from dataLoader import dataLoader as dl
from RBM import RBM


#%%
# 1. Declaring RBM parameters

n = 50 # the number of hidden units
m = 27613 # the max number of visible units
k = 2 # number of possible values for V
T = 1 # number of gibbs sampling cycles

batchSize = 1000 # number of samples per batch
epochs = 50 # how many epochs to train for
alpha = 0.01/batchSize # learning rate alpha

#%%
# 2. Loading the training and validation data

trainData = dl.loadData("trainData.txt")
trainMap = dl.loadData("trainMap.txt")
#valData = loadtxt("valData.txt", comments="#", delimiter=" ", dtype =int, unpack=False)

#%%

# 3. initialize and train the model

model = RBM(k,m,n)


#%%
count = 0
for listElem in trainData:
    count += len(listElem)                    
print('Total Number of elements : ', count)