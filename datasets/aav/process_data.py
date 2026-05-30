import numpy as np
import pandas as pd
import os

os.chdir('/home/apa2237/generative_model_work/datasets/aav')

# Load data
print("Loading CSV...")
df = pd.read_csv('full_data.csv', low_memory=False)
print(f"Loaded {len(df)} rows")

# Create data array
print("Creating data arrays...")
data = np.array([[row['full_aa_sequence'], row['score']] for _, row in df.iterrows()], dtype=object)

# Make dict
data_dict = {data[i][0]: data[i][1] for i in range(len(data))}
np.save('data_dict.npy', data_dict)
np.save('data.npy', data)

# Find max length
max_len = max(len(d[0]) for d in data)
print(f"Max sequence length: {max_len}")

# Amino acids
amino_acid = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
np.save('categorical_variables', amino_acid)

def onehotseq(sequence):
    seq_len = len(sequence)
    seq_en = np.zeros((seq_len, len(amino_acid)))
    aa_seq = ''
    act_len = 0
    for i in range(seq_len):
        if sequence[i] in amino_acid:
            pos = amino_acid.index(sequence[i])
            seq_en[act_len, pos] = 1
            act_len += 1
            aa_seq += sequence[i]
    return seq_en[0:act_len, :], aa_seq

# One-hot encode
print("One-hot encoding sequences...")
ohe = np.zeros((data.shape[0], max_len, len(amino_acid)))
seq_string = np.zeros((data.shape[0],), dtype=object)
seq_lengths = np.zeros((data.shape[0],))
scores = np.zeros((data.shape[0], 1))

for i in range(len(data)):
    if i % 50000 == 0:
        print(f"  Processing {i}/{len(data)}")
    seq_en, aa_seq = onehotseq(data[i, 0])
    seq_string[i] = aa_seq
    ohe[i, 0:seq_en.shape[0], :] = seq_en
    seq_lengths[i] = seq_en.shape[0]
    scores[i, 0] = data[i, 1]

# Normalize scores to [0, 1]
score_min, score_max = scores.min(), scores.max()
output_y = (scores - score_min) / (score_max - score_min)
print(f"Scores normalized from [{score_min:.3f}, {score_max:.3f}] to [0, 1]")

# Randomly select subsets: 5000 train, 500 valid, 500 test
np.random.seed(42)
all_idx = np.arange(len(df))
np.random.shuffle(all_idx)

x_train = all_idx[:5000]
x_valid = all_idx[5000:5500]
x_test = all_idx[5500:6000]

print(f"Train: {len(x_train)}, Valid: {len(x_valid)}, Test: {len(x_test)}")

# Save files
print("Saving files...")
np.save('./x_train', ohe[x_train])
np.save('./len_train', seq_lengths[x_train])
np.save('./y_train', output_y[x_train])
np.save('./seq_train', seq_string[x_train])

np.save('./x_valid', ohe[x_valid])
np.save('./len_valid', seq_lengths[x_valid])
np.save('./y_valid', output_y[x_valid])
np.save('./seq_valid', seq_string[x_valid])

np.save('./x_test', ohe[x_test])
np.save('./len_test', seq_lengths[x_test])
np.save('./y_test', output_y[x_test])
np.save('./seq_test', seq_string[x_test])

print("\nDone! Final shapes:")
print(f"x_train: {ohe[x_train].shape}")
print(f"x_valid: {ohe[x_valid].shape}")
print(f"x_test: {ohe[x_test].shape}")
