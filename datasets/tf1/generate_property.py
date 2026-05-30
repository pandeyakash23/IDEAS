import numpy as np
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

amino_acid = ['A', 'G','C','T'] # DNA nucleotides

class output_property_oracle():
    def __init__(self):
        self.repo = np.load(os.path.join(SCRIPT_DIR, 'data_dict.npy'), allow_pickle=True).tolist()

    def output_property(self, sequences):
        l = len(sequences)
        prop = np.zeros((l,))

        for i,seq in enumerate(sequences):
            if seq in self.repo :
                prop[i] = self.repo[seq]
            else:
                prop[i] = 0  # or some default value if sequence not found
        return prop

def onehotseq(sequence):
    seq_len = len(sequence)
    seq_en = np.zeros(( seq_len, np.shape(amino_acid)[0]))
    for i in range(seq_len):
        pos = amino_acid.index(sequence[i])
        seq_en[i,pos] = 1
    return seq_en
