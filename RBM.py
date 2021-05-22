#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:00:14 2021

Author: Juan Rios

This file contains the RBM class, which handles construction, training and classification of the rbm.
"""

import numpy as np


class RBM:
    
  def __init__(self, k,m,n):
      
    self.n = n
    self.m = m
    
    self.c = 2*np.random.rand(n)-1
    self.b = 2*np.random.rand(m)-1
  
    self.heightW = k * n
    self.dW = np.zeros((self.heightW,m))
    self.W = 2*np.random.rand(self.heightW,m) - 1
      
      
