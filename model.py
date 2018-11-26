import keras as k
from keras.models import Sequential, Model
from keras.layers import Activation, BatchNormalization, Input, Dense
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras import losses
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import argmax
from numpy import ndarray
import tensorflow as tf
import math
import re
import string
from collections import OrderedDict
import pandas as pd
import csv
from collections import defaultdict

"""DATA PREPROCESSING"""
with open('1K_unformated_unlabeled_SMILES_data.txt') as my_file:
    dataset = my_file.readlines()

with open('test_unformated_unlabeled_SMILES_data.txt') as my_file:
    test_dataset = my_file.readlines()

#Formating data into array of arrays
array_of_array = []

for line in dataset:
        formated_data = line.split()
        array_of_array.append(formated_data)

test_array_of_array = []

for test_line in test_dataset:
        test_formated_data = line.split()
        test_array_of_array.append(test_formated_data)

#Extracting the SMILES strings from excel dataset
formated_SMILE_array = []

c_reader = csv.reader(open('1K_unformated_unlabeled_SMILES_data.txt', 'r'), delimiter=';')
col_2 = [x[0] for x in c_reader]
for string in col_2:
        head, sep, tail = string.partition('\t')
        SMILE = tail
        formated_SMILE_array.append(SMILE)

test_formated_SMILE_array = []

test_c_reader = csv.reader(open('test_unformated_unlabeled_SMILES_data.txt', 'r'), delimiter=';')
test_col_2 = [x[0] for x in test_c_reader]
for test_string in test_col_2:
        head, sep, tail = test_string.partition('\t')
        test_SMILE = tail
        test_formated_SMILE_array.append(SMILE)

#Extracting the name of the Molecule
for data_array in array_of_array:
        for info in data_array:
                name_ind_position = data_array[2]

for test_data_array in test_array_of_array:
        for test_info in test_data_array:
                test_name_ind_position = test_data_array[2]
        
#Extracting the solubility levels
solubility_array = []

for data_array in array_of_array:
        solubility_ind_position = data_array[3]
        solubility_array.append(solubility_ind_position)

test_solubility_array = []
for test_data_array in test_array_of_array:
        test_solubility_ind_position = test_data_array[3]
        test_solubility_array.append(test_solubility_ind_position)

#Turning SMILES into Explicit Bit Vectors (RDKit prefered format)
mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in formated_SMILE_array]

test_mols = [Chem.rdmolfiles.MolFromSmiles(test_SMILES_string) for test_SMILES_string in test_formated_SMILE_array]

#Convert training molecules into training fingerprints
bi = {}
fingerprints = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=256) for m in mols]

test_bi = {}
test_fingerprints = [rdMolDescriptors.GetMorganFingerprintAsBitVect(test_m, radius=2, bitInfo= bi, nBits=256) for test_m in test_mols]

#Convert training fingerprints into binary, and put all training binaries into arrays
fingerprints_array = []
for fingerprint in fingerprints:
        array = np.zeros((1,), dtype= int)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        fingerprints_array.append(array)

test_fingerprints_array = []
for test_fingerprint in test_fingerprints:
        test_array = np.zeros((1,), dtype= int)
        DataStructs.ConvertToNumpyArray(test_fingerprint, test_array)
        test_fingerprints_array.append(test_array)

#print (fingerprints_array, solubility_array)
"""NEURAL NETWORK"""
#The neural network model
model = Sequential([
    Dense(256, input_shape=(256,), activation= "relu"),
    Dense(128, activation= "tanh"),
    Dense(64, activation= "tanh"),
    Dense(34, activation= "tanh"),
    Dense(16, activation= "tanh"),
    BatchNormalization(axis=1),
    Dense(1, activation= "tanh")
])
model.summary()

#Compiling the model
model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=0.0005), metrics=["accuracy"])

#Training the model
model.fit(np.array(fingerprints_array), np.array(solubility_array), validation_split=0.1, batch_size=5, epochs= 200, shuffle=True, verbose=1)

#Predictions with test dataset
predictions = model.predict(np.array(test_fingerprints_array), batch_size=1, verbose=1)

#Predictions with test dataset
predictions = model.predict(np.array(test_fingerprints_array), batch_size=1, verbose=1)

for prediction in predictions:
    print (prediction)