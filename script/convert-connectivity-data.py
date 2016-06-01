"""Script to convert connectivity data from Varshney et al to Numpy format.

Data downloaded from:
http://www.ifp.illinois.edu/~varshney/elegans/

ConnOrdered_040903.mat:
http://www.ifp.illinois.edu/~varshney/elegans/ConnOrdered_040903.mat

NeuronTypeOrdered_040903.mat
http://www.ifp.illinois.edu/~varshney/elegans/NeuronTypeOrdered_040903.mat

Reading code (including data manipulation) based on:
http://www.ifp.illinois.edu/~varshney/elegans/datareader.m
"""

import numpy as np
from scipy import io

conn = io.loadmat('ConnOrdered_040903.mat')
neuron = io.loadmat('NeuronTypeOrdered_040903.mat')

# Neuron type information is encoded in the array of three-letter strings,
# `neuron['NeuronType_ordered']`. The final letter in this array is one of
# S, I, or M, for Sensory neuron, Interneuron, or Motoneuron. This is the only
# information we are after, so we encode these as 0, 1, and 2 in a numpy array.

typedict = {'S': 0, 'I': 1, 'M': 2}
neurontypearray = np.array([typedict[ntype[0][0][-1]]
                            for ntype in neuron['NeuronType_ordered']])
np.save('neuron-types.npy', neurontypearray.astype(np.int8))

# the gap network has three spurious self-loops that must be manually deleted,
# see code in http://www.ifp.illinois.edu/~varshney/elegans/datareader.m
# Note that we must use 0-indexing instead of Matlab's 1-indexing.

gap_network = conn['Ag_t_ordered']  # CSC matrix
gap_network[94, 94] = 0
gap_network[106, 106] = 0
gap_network[216, 216] = 0

np.save('gap-network.npy', gap_network.toarray().astype(np.float32))

# finally, the simplest of the three, the chemical synapse network:

chem_network = conn['A_init_t_ordered']  # CSC matrix
np.save('chem-network.npy', chem_network.toarray().astype(np.float32))

# We also need the neuron labels as a list

labels = conn['Neuron_ordered']
neurons = np.array([label[0][0] for label in labels])
np.save('neurons.npy', neurons)

