# -*- coding: utf-8 -*-
"""
Author: Juan Rios

This file loads and formats the data from Data.csv. 
The data is formatted into the training and validation data used on the RBM. 
The data is stored into text files trainData.txt and trainMap.txt. The
validation data is stored into valData.txt
"""


import pandas as pd
import csv


# 1. Load the raw data into a dataframe, convert to list

data = pd.read_csv('Data.csv', delimiter=',')
dataList = data.to_numpy()

# 2. determine the number of users

trainRatio = 0.8 # 80% of total dat is train
numUsers = max(dataList[:,1]) + 1
numTrain = round(len(dataList)*trainRatio)
numVal = len(dataList) - numTrain

# 3. Allocate space for training data
trainData = [ [] for _ in range(numUsers) ]
trainMap = [ [] for _ in range(numUsers) ]

# 4. Create the training data
counter  = 0
while (counter < numTrain):
    
    lst = dataList[counter]
    
    user = lst[1]
    q = lst[0]
    answer = lst[3]
    
    trainData[user].append(answer)
    trainMap[user].append(q)
    
    
    if (counter % 100000 == 0):
        print(counter)
    
    counter  =  counter + 1;
    
# 5. Print the train data into a txt file    
    
with open("trainData.txt","w") as f:
    wr = csv.writer(f, delimiter = " ")
    wr.writerows(trainData)
    
with open("trainMap.txt","w") as g:
    wr = csv.writer(g, delimiter = " ")
    wr.writerows(trainMap)
    
# 6. Print the validation data into a txt file

with open("valData.txt","w") as h:
    wr = csv.writer(h, delimiter = " ")
    wr.writerows(dataList[numTrain:])    
    
from numpy import loadtxt
valData= loadtxt("valData.txt", comments="#", delimiter=" ", dtype =int, unpack=False)
    