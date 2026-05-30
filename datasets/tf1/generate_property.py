import numpy as np
amino_acid = ['A', 'G','C','T'] # O is the uncommon amino acid, so total length is 21

class output_property_oracle():
    def __init__(self):
        self.repo = np.load("/home/apa2237/generative_model_work/datasets/tf_ARX_L343Q_R1_8mers/data_dict.npy", allow_pickle=True).tolist()
        # print(self.repo)
        # self.repo = list(self.repo)

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
        # if sequence[i] in amino_acid:
        pos = amino_acid.index(sequence[i])
        seq_en[i,pos] = 1    
        # elif (sequence[i] not in amino_acid):
        #     pos = amino_acid.index('X')
        #     seq_en[act_len,pos] = 1
        #     act_len += 1 
    return seq_en