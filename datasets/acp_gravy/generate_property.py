import numpy as np
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

## different amino acids
amino_acid = ['A', 'V', 'F', 'I', 'L','D','E','K','S','T','Y','C','N','Q', 'P','M', 'R', 'H', 'W', 'G']# X is the uncommon amino acid, so total length is 6
np.save(os.path.join(SCRIPT_DIR, 'categorical_variables.npy'), amino_acid)
hydro = [1.8, 4.2, 2.8, 4.5, 3.8, -3.5, -3.5, -3.9, -0.8, -0.7, -1.3, 2.5, -3.5, -3.5, -1.6, 1.9, -4.5, -3.2, -0.9, -0.4]

def output_property(x):
    N = len(x)
    prop = np.zeros((N,))
    # x = np.argmax(x, axis=-1)
    for i in range(N):
        x[i] = x[i].split('0')[0]
        l = len(x[i])
        x_sample = x[i][0:l]
        sample_prop = 0
        for j in  range(l):
            sample_prop += hydro[amino_acid.index(x_sample[j])]
        
        prop[i] = sample_prop/(l+1E-18)
    return prop

def output_property_gfn(x):
    N = len(x)
    prop = np.zeros((N,))
    # x = np.argmax(x, axis=-1)
    for i in range(N):
        x[i] = x[i].split('%')[0]
        l = len(x[i])
        x_sample = x[i][0:l]
        sample_prop = 0
        for j in  range(l):
            sample_prop += hydro[amino_acid.index(x_sample[j])]
        
        prop[i] = sample_prop/l
    return prop


def onehotseq(sequence):
    seq_len = len(sequence)
    seq_en = np.zeros(( seq_len, np.shape(amino_acid)[0]))
    for i in range(seq_len):
        # if sequence[i] in amino_acid:
        pos = amino_acid.index(sequence[i])
        seq_en[i,pos] = 1    
        # elif (sequence[i] not in amino_acid):
        #     pos = amino_acid.index('X')
        #     seq_en[act_len,pos] = 1
        #     act_len += 1 
    return seq_en