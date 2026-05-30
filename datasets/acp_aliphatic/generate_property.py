import numpy as np
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

## different amino acids
amino_acid = ['A', 'V', 'F', 'I', 'L', 'D', 'E', 'K', 'S', 'T', 'Y', 'C', 'N', 'Q', 'P', 'M', 'R', 'H', 'W', 'G']
np.save(os.path.join(SCRIPT_DIR, 'categorical_variables.npy'), amino_acid)

# Aliphatic index coefficients
# AI = 100 * (Ala% + 2.9*Val% + 3.9*(Ile% + Leu%))
aliphatic_coef = {
    'A': 1.0,    # Alanine
    'V': 2.9,    # Valine
    'I': 3.9,    # Isoleucine
    'L': 3.9,    # Leucine
}


def output_property(x):
    """
    Calculate aliphatic index for a list of sequences.
    Aliphatic index = 100 * (Ala% + 2.9*Val% + 3.9*(Ile% + Leu%))
    """
    N = len(x)
    prop = np.zeros((N,))
    for i in range(N):
        seq = x[i].split('0')[0]
        l = len(seq)
        if l == 0:
            prop[i] = 0
            continue

        # Count aliphatic amino acids
        aliphatic_sum = 0
        for aa in seq:
            if aa in aliphatic_coef:
                aliphatic_sum += aliphatic_coef[aa]

        # Aliphatic index formula
        prop[i] = 100 * (aliphatic_sum / l)
    return prop


def output_property_gfn(x):
    """
    Calculate aliphatic index for GFN-style sequences (split by %).
    """
    N = len(x)
    prop = np.zeros((N,))
    for i in range(N):
        seq = x[i].split('%')[0]
        l = len(seq)
        if l == 0:
            prop[i] = 0
            continue

        # Count aliphatic amino acids
        aliphatic_sum = 0
        for aa in seq:
            if aa in aliphatic_coef:
                aliphatic_sum += aliphatic_coef[aa]

        # Aliphatic index formula
        prop[i] = 100 * (aliphatic_sum / l)
    return prop


def onehotseq(sequence):
    """Convert a sequence string to one-hot encoding."""
    seq_len = len(sequence)
    seq_en = np.zeros((seq_len, np.shape(amino_acid)[0]))
    for i in range(seq_len):
        pos = amino_acid.index(sequence[i])
        seq_en[i, pos] = 1
    return seq_en
