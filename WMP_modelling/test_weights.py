import pandas as pd
import numpy as np

weights1 = pd.read_csv('C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access '
                       'Model\\WMP\\Semantic representation simulations\\weights_512epoch_no_noise.csv')
weights2 = pd.read_csv('C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access '
                       'Model\\WMP\\replication\\rodd2004_dom_effect\\dom_effect_weights.csv')
weights3 = pd.read_csv('C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access '
                       'Model\\WMP\\replication\\rodd2004_dom_effect\\dom_effect_test_weights.csv')
weights4 = pd.read_csv('C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access '
                       'Model\\WMP\\replication\\rodd2004_dom_effect\\dom_effect_test2_weights.csv')

weights1 = np.array(weights1.iloc[:, 1:])
weights2 = np.array(weights2.iloc[:, 1:])
weights3 = np.array(weights3.iloc[:, 1:])
weights4 = np.array(weights4.iloc[:, 1:])

average_synapse_weights1 = []
for unit in range(len(weights1[:, 0])):
    average_synapse_weights1.append(weights1[unit, :].max())

average_synapse_weights2 = []
for unit in range(len(weights2[:, 0])):
    average_synapse_weights2.append(weights2[unit, :].max())

average_synapse_weights3 = []
for unit in range(len(weights3[:, 0])):
    average_synapse_weights3.append(weights3[unit, :].max())

average_synapse_weights4 = []
for unit in range(len(weights4[:, 0])):
    average_synapse_weights4.append(weights4[unit, :].max())

print(max(average_synapse_weights1))
print(max(average_synapse_weights2))
print(max(average_synapse_weights3))
print(max(average_synapse_weights4))
breakpoint()
