# Restricted-Boltzmann-Machines
This repository contains files for a project where a Restricted Boltzmann Machine is trained and used to make predictions on an existing data set.
This README gives a description of the files in this project.

1. Main File(s) <br />
  1. RestrictedBoltzmannMachine.pdf - This PDF describes the entire project: From the mathematical background, the training, and the application of the RBM to the       data.<br />
2. Source File(s) <br />
  1. dataFormat.py - this file loads the raw data from Data.csv, and outputs the training data as "trainData.txt", and "trainMap.txt". Additionally, the validation       data is outputted as "valData.txt" <br />
  2. dataLoader.py - this file contains the loadData() method that loads a text file into a list of lists. <br />
  3. init.py - this file is the entry point of the program. The hyper parameters are defined, the data is loaded, the RBM is initiated, trainedm, and evaluated <b/>
  4. RBM.py - this file is the class file for the RBM, which initiates the data structures required to hold the RBM weightsm biases, and other parameters, along with methods that train and evaluate the RBM
3. Data Files(s)
  1. N/A
5. References<br />
  1. NeurIPS 2020 Challenge.pdf - this PDF contains the origin of the data set and details the challenged that was posed by NeurIPS in 2020.<br />
  2. Intro_to_RBM_Fischer_Ingel - this PDF contains a nice intro to RBMs<br />
