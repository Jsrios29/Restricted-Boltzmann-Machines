#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:00:14 2021

Author: Juan Rios

This file contains the RBM class, which handles construction, training and 
classification of the rbm.
"""

import numpy as np
import math


class RBM:
    
  def __init__(self, k,m,n, alpha):
      
    self.n = n # number of hidden nodes
    self.m = m # number of possible visible units
    
    self.c = 2*np.random.rand(n)-1 # randomly initialized bias values for hidden nodes
    self.b = 2*np.random.rand(m)-1 # randomly inti bias values for the visible node
  
   # self.heightW = k * n # computes the number of rows for W
   # self.dW = np.zeros((self.heightW,m)) # initiates an array of 0s to hold the average dw for eacdh weight
   # self.dWCounts = np.zeros((self.heightW,m)) # this array keeps track of how many times each dw is updates
   # self.W = 2*np.random.rand(self.heightW,m) - 1 # randomly initialize the weights matrix W
    self.alpha = alpha # learning rate  
    
    self.dW = np.zeros((n,m))
    self.W = 2*np.random.rand(n,m) - 1 # EXPRRIMENT: only care about k =1, rather than k=0, k=1
 
  # This method trains the RBM by updating the RBM's W matrix
  #
  # numEpochs - number of training epochs
  # numBatches - number of training batches per epoch
  # batchSize - number of training samples per batch
  # trainData - the student responses of 1 and 0
  # trainMap - the correponsinding questions to the trainData
  # T - the number of gibb sampling rounds  
  def train(self, numEpochs, numBatches, batchSize, trainData, trainMap, T):
      
      # 1. get the number of samples in the training data
      numSamples = len(trainData)
      # 2. train the rbm  numEpochs number of times
      for epoch in range(numEpochs):
          
          # 3. obtain a random ordering of the training samples
          sampleQueue = np.random.permutation(numSamples)
          sampleCount = 1
          
          # 4. for each random sample
          for sampleId  in sampleQueue:
              
             # 5. pick the random sample
             sample = trainData[sampleId]
             sampleMap = trainMap[sampleId]
             
             # 6. Obtain V after T gibb sampling iterations
             VGibbs = self.sampleGibbs(sample, sampleMap, T)
             
             # 7. Update dW
             self.updateDW(sample, sampleMap, VGibbs)
             print(sampleCount)
             
             # 8. check if end of batch
             if ((sampleCount % batchSize == 0) or ( sampleCount == numSamples)):
                 
                 # 9. Update W after every batch
                 self.updateW()
                 # 10. reset dW and dWCounts
                 self.dW = np.zeros((self.heightW, self.m))
                 self.dWCounts = np.zeros((self.heightW, self.m))
                 
             sampleCount = sampleCount + 1
             
  # This method samples a binary vector H given a binary vector V 
  # binary is a flag that if true produces a binary vector (0s and 1s),
  # if false then it gives out the probability of each element in the vector
  # to be 1          
  def sampleH(self, V, vMap, binary):
      
      H = [None]*self.n  
      
      wj = np.transpose(self.W[:, vMap])
      tot = np.dot(V, wj)
      tot = self.c + tot
      
      H = self.sigmoid(tot)
      
      if (binary):
          H = np.round(H)
      
      return H
       
  
  # This method finds probability that it is 1 by default instead of 0   
  # which means using the lower half of W   
  def sampleV(self, H, V, vMap):
      
      wi = self.W[:,vMap]
      tot = np.dot(H,wi)
      tot = self.b[vMap] + tot
      V =  np.round(self.sigmoid(tot))
      return V
               
      
  def sampleGibbs(self, sample, sampleMap, T):
      
      V = sample.copy()
      
      for rnd in range(T):
          
          
          H = self.sampleH( V, sampleMap, True)
          V = self.sampleV(H, V, sampleMap)
          
      return V 
             
  def updateDW(self, sample, sampleMap, VGibbs):
      
      pH0 = self.sampleH(sample, sampleMap, False)
      pHgibbs = self.sampleH(VGibbs, sampleMap, False)
      
      for v in range(len(sample)):
          
          qId = sampleMap[v];
          
          for h in range(self.n):
              
              vk = 0;
              if (sample[v] == VGibbs[v]):
                  vk = 1
             
              self.dW[self.n*sample[v] + h ,qId] = self.dW[self.n*sample[v] + h ,qId] + \
                  (pH0[h] - pHgibbs[h]*vk)*self.alpha 
                  
              self.dWCounts[self.n*sample[v] + h ,qId] = self.dWCounts[self.n*sample[v] + h ,qId] + 1
                  
  def updateW(self):
      
      #meanDW = np.divide(self.dW, self.dWCounts)
      self.W = self.W + self.dW
      
  def validate(self, valData, trainData, trainMap):
      
      valSize = len(valData)
      rightCounter = 0;
      for vdat in valData:
          
          qId = vdat[0]
          studentId = vdat[1]
          label = vdat[3]
          
          sample = trainData[studentId]
          sampleMap = trainMap[studentId]
          
          V = sample.copy()
          
          H = self.sampleH(V, sampleMap, False)
          
    
          tot = np.dot(H, self.W[self.n:, qId])
          pred = round(self.sigmoid(self.b[qId] + tot))
      
          if pred == label:
              rightCounter = rightCounter + 1
              
      print('Prediction rate = ' + str(rightCounter/valSize))  
      
      
  @staticmethod
  def sigmoid(z):            
      return 1/(1 + np.exp(-z))
      
      
      
      
      
      
      
      
