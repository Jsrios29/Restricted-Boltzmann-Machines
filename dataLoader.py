"""
Created on Sat May 22 15:46:22 2021

Author: Juan Rios

This file contains a class that helps loading the training data
"""

class dataLoader:
    
  @staticmethod
  def loadData(fileName):
      
      content = []
      with open(fileName) as f:
          for line in f:
              data = line.split()
              data = [int(i) for i in data]
              content.append(data)
          
    
      return content
