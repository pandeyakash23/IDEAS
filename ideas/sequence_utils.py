"""Utility functions for manipulating sequences."""
import random
from typing import List, Union
import math
import time

import numpy as np

AAS = "ILVAGMFYWEDQNHCRKSTP"
"""str: Amino acid alphabet for proteins (length 20 - no stop codon)."""

RNAA = "UGCA"
"""str: RNA alphabet (4 base pairs)."""

DNAA = "TGCA"
"""str: DNA alphabet (4 base pairs)."""

BA = "01"
"""str: Binary alphabet '01'."""


def construct_mutant_from_sample(
    pwm_sample: np.ndarray, one_hot_base: np.ndarray
) -> np.ndarray:
    """Return one hot mutant, a utility function for some explorers."""
    one_hot = np.zeros(one_hot_base.shape)
    one_hot += one_hot_base
    i, j = np.nonzero(pwm_sample)  # this can be problematic for non-positive fitnesses
    one_hot[i, :] = 0
    one_hot[i, j] = 1
    return one_hot


def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:
    """
    Return the one-hot representation of a sequence string according to an alphabet.

    Args:
        sequence: Sequence string to convert to one_hot representation.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        One-hot numpy array of shape `(len(sequence), len(alphabet))`.

    """
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


def one_hot_to_string(
    one_hot: Union[List[List[int]], np.ndarray], alphabet: str
) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.

    Args:
        one_hot: One-hot of shape `(len(sequence), len(alphabet)` representing
            a sequence.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        Sequence string representation of `one_hot`.

    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])


def generate_single_mutants(wt: str, alphabet: str) -> List[str]:
    """Generate all single mutants of `wt`."""
    sequences = [wt]
    for i in range(len(wt)):
        tmp = list(wt)
        for j in range(len(alphabet)):
            tmp[i] = alphabet[j]
            sequences.append("".join(tmp))
    return sequences


def generate_random_sequences(length: int, number: int, alphabet: str) -> List[str]:
    """Generate random sequences of particular length."""
    return [
        "".join([random.choice(alphabet) for _ in range(length)]) for _ in range(number)
    ]


def generate_random_mutant(sequence: str, mu: float, alphabet: str) -> str:
    """
    Generate a mutant of `sequence` where each residue mutates with probability `mu`.

    So the expected value of the total number of mutations is `len(sequence) * mu`.

    Args:
        sequence: Sequence that will be mutated from.
        mu: Probability of mutation per residue.
        alphabet: Alphabet string.

    Returns:
        Mutant sequence string.

    """
    count_mut = 0
    mutant = []
    for s in sequence:
        if (random.random() < mu):
            # print('+1')
            mutant.append(random.choice(alphabet))
            count_mut += 1
        else:
            mutant.append(s)
    return "".join(mutant)

def generate_random_multiple_mutant(sequence: str, mu: float, alphabet: str, num_pos: int) -> str:
    """
    Generate a mutant of `sequence` where each residue mutates with probability `mu`.

    So the expected value of the total number of mutations is `len(sequence) * mu`.

    Args:
        sequence: Sequence that will be mutated from.
        mu: Probability of mutation per residue.
        alphabet: Alphabet string.

    Returns:
        Mutant sequence string.

    """
    l = len(sequence)
    mut_pos = random.sample(range(l), num_pos)
    count_mut = 0
    mutant = []
    for c_id, s in enumerate(sequence):
        if c_id in mut_pos:
            # print('+1')
            mutant.append(random.choice(alphabet))
            count_mut += 1
        else:
            mutant.append(s)
    
    # print('===================')
    # print(sequence)
    # print('Mutant', mutant) 
    # print('===================')
    return "".join(mutant)


def generate_importance_based_mutant(sequence: str, imp: list,temp: float, alphabet: str, imp_aa: list) -> str:
    """
    Generate a mutant of `sequence` where each residue mutates baed
    on the importance scores in imp.

    Args:
        sequence: Sequence that will be mutated from.
        imp: contains the importance of each position in
        the primary sequence.
        temp: temp variable to control the importance spread.
        alphabet: Alphabet string.

    Returns:
        Mutant sequence string.

    """
    imp = imp/np.max(imp)
    # imp = -imp - 1
    imp = imp + 1
    mu = 1/(imp)
    
    
    ## softmax #####
    mu = np.exp(mu/temp)
    mu = mu/np.sum(mu)
    
    ##########################################   
    # max_val = max(mu)
    #############################################
    count_mut = 0
    mutant = []
    for i, s in enumerate(sequence):
        # if (random.random() < mu[i]):
        if (random.random() < mu[i]) and (count_mut==0):
            
            ## including temp in imp_aa
            imp_aa = np.array(imp_aa)
            imp_aa = np.exp(imp_aa/temp)
            imp_aa = imp_aa/np.sum(imp_aa)
            imp_aa = imp_aa.tolist()
            ## choosing based on weights
            choice_aa = random.choices(alphabet, weights=imp_aa)[0]
            
            # ## top k-mer
            # top_idx_5 = np.argsort(np.array(imp_aa))[::-1]
            # top_idx_5 = top_idx_5[0:5]
            # choice_aa = random.choices(top_idx_5)[0]
            # choice_aa = alphabet[choice_aa]
            
            ## choosing randomly
            # choice_aa = random.choices(alphabet)[0]
         
            
            mutant.append(choice_aa)
            count_mut += 1
        else:
            mutant.append(s)
    return "".join(mutant)

def temperature_scaled_softmax(imp, temp):
    imp = np.exp(imp/temp)
    imp = imp/(np.sum(imp)+1E-18)
    return imp

def give_informed_random(imp,temp,good_favor):
    if good_favor:
        node_badness = imp
    else:
        node_badness = -imp #1/(imp+1E-18)
    all_motif_pos = np.arange(len(node_badness), dtype=int)
    
    temp_badness = temperature_scaled_softmax(node_badness,temp)
    inform_idx = random.choices(all_motif_pos,weights=temp_badness)[0]
    all_motif_pos = np.delete(all_motif_pos, inform_idx)
    
    randomizing = temperature_scaled_softmax(node_badness,1000) #CHANGE
    # randomizing = temperature_scaled_softmax(node_badness,temp) #CHANGE

    randomizing = randomizing[all_motif_pos]

    random_idx = random.choices(all_motif_pos,weights=randomizing)[0]
    # random_idx = random.choices(all_motif_pos)[0]

    return inform_idx, random_idx


def motif_level_mutation(sequence: str, motif_mean_dict: dict,node_motifs: dict, motif_size:int, temp: float) -> str:
    """
    This function performs mutations at the motif-level

    Args:
        sequence: Sequence that will be mutated from.
        motif_mean_dict: dict that contains motifs and their importance
        node_motifs: dict that contains motifs and their importance for 
        the given sequence
        motif_size: size of the motif that we are considering
        temp: for doing temp_scaling in the softmax function

    Returns:
        Mutant sequence string.
    """
    # print(sequence)
    node_motif_values = np.array(list(node_motifs.values()))
    m_inform, m_random = give_informed_random(node_motif_values,temp,good_favor=False)
    
    candidate_values = np.array(list(motif_mean_dict.values()))    
    check_0 = True
    while check_0:
        with_inform, with_random = give_informed_random(candidate_values,temp,good_favor=True)
        candidate_keys = list(motif_mean_dict.keys())
        with_inform, with_random = candidate_keys[with_inform], candidate_keys[with_random]
        
        if ('0' not in list(with_inform)) and ('0' not in list(with_random)):
            check_0 = False            
        # else:
        #     print(with_inform, with_random)
    
    # print(sequence[m_inform:m_inform+motif_size],sequence[m_random:m_random+motif_size])
    
    sequence = list(sequence)
    
    sequence[m_inform:m_inform+motif_size] = list(with_inform)
    sequence[m_random:m_random+motif_size] = list(with_random) #CHANGE
    
    sequence = ''.join(sequence)
    return sequence

    # print(sequence)
    # print(Aaaaa)
    # # node_badness = 1/node_motif_values
    # # node_badness = np.exp(node_badness/temp)
    # # node_badness = node_badness/np.sum(node_badness)
    
    # # all_motif_pos = np.arange(len(node_badness), dtype=int)
    # # inform_idx = random.choices(all_motif_pos,weights=node_badness)[0]
    # # all_motif_pos = np.delete(all_motif_pos, inform_idx)
    # # random_idx = random.choice(all_motif_pos)
    # # print(inform_idx, random_idx)
    
    # imp = imp/np.max(imp)
    # # imp = -imp - 1
    # imp = imp + 1
    # mu = 1/(imp)
    
    
    # ## softmax #####
    # mu = np.exp(mu/temp)
    # mu = mu/np.sum(mu)
    
    # ##########################################   
    # # max_val = max(mu)
    # #############################################
    # count_mut = 0
    # mutant = []
    # for i, s in enumerate(sequence):
    #     # if (random.random() < mu[i]):
    #     if (random.random() < mu[i]) and (count_mut==0):
            
    #         ## choosing based on weights
    #         choice_aa = random.choices(alphabet, weights=imp_aa)[0]
            
    #         # ## top k-mer
    #         # top_idx_5 = np.argsort(np.array(imp_aa))[::-1]
    #         # top_idx_5 = top_idx_5[0:5]
    #         # choice_aa = random.choices(top_idx_5)[0]
    #         # choice_aa = alphabet[choice_aa]
            
    #         ## choosing randomly
    #         # choice_aa = random.choices(alphabet)[0]
         
            
    #         mutant.append(choice_aa)
    #         count_mut += 1
    #     else:
    #         mutant.append(s)
    # return "".join(mutant)
 
