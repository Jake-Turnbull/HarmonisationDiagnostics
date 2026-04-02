#%%
from DiagnoseHarmonisation import DiagnosticFunctions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

group1 = np.array([1,2,3,4,5])
group2 = np.array([1,2,3,4,5])
group = np.concatenate((group1, group2))
batch = np.array([0,0,0,0,0,1,1,1,1,1])
# Convert batch to numeric codes

#%%

def test_cohens_d():
    # Create a single vector with array with 3 groups:
    data = np.random.rand(100,10)  # 100 samples, 10 features
    batch = np.array([0]*30 + [1]*40 + [2]*30)  # 3 batches
    
    # Add a binary batch effect to the data 
    data[batch == 1] += 0.5  # Shift batch 1
    data[batch == 2] += 1.0  # Shift batch 2

    a,b = DiagnosticFunctions.Cohens_D(data, batch)
    # Output: 
    # np.ndarray: Cohen's d values, shape = (num_pairs, num_features).
    #list: Pair labels, each as a tuple of (name1, name2).
    assert type(a) == np.ndarray
    assert type(b) == list
    assert np.size(a) == 30  # 3 pairs for each of the 10 features and each batch to average



