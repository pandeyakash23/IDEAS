"""Defines the Adalead explorer class."""
import random
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict

import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the IDEAS root directory (two levels up from ideas/aav/)
IDEAS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Add paths relative to IDEAS root
sys.path.append(os.path.join(SCRIPT_DIR, '..'))  # for sequence_utils
sys.path.append(os.path.join(IDEAS_ROOT, 'datasets', 'aav'))
sys.path.append(os.path.join(IDEAS_ROOT, 'importance_models', 'aav'))

import sequence_utils as s_utils
from contributions_score_generative import contribution_score, predict_property, train_importance_model

# Data path relative to IDEAS root
data_path = os.path.join(IDEAS_ROOT, 'datasets', 'aav')

alphabets_of_color = np.load(os.path.join(data_path, 'categorical_variables.npy'), allow_pickle=True)
alphabets_of_color = alphabets_of_color.tolist()
# Pre-compute alphabet index mapping for O(1) lookup
aa_to_idx = {aa: i for i, aa in enumerate(alphabets_of_color)}

def onehotseq(sequence):
    seq_len = len(sequence)
    seq_en = np.zeros((seq_len, len(alphabets_of_color)))
    for i in range(seq_len):
        seq_en[i, aa_to_idx[sequence[i]]] = 1
    return seq_en

class Adalead():
    """
    Adalead explorer.

    Algorithm works as follows:
        Initialize set of top sequences whose fitnesses are at least
            (1 - threshold) of the maximum fitness so far
        While we can still make model queries in this batch
            Recombine top sequences and append to parents
            Rollout from parents and append to mutants.

    Partial code usage from:
        Sinai et al., "AdaLead: A simple and robust adaptive greedy search algorithm
        for sequence design", https://arxiv.org/abs/2010.02141

    """

    def __init__(
        self,
        model,
        model_args,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        mu: int = 1,
        recomb_rate: float = 0,
        threshold: float = 0.05,
        rho: int = 0,
        eval_batch_size: int = 20,
        model_contri: Optional[type] = None,
        criterion_contri: Optional[type] = None,
        optimizer_contri: Optional[type] = None,
        log_file: Optional[str] = None,
        motif_size: int = 1,
        motif_based: bool = False,
        initial_ohe: Optional[np.ndarray] = None,
        initial_y: Optional[np.ndarray] = None,
        initial_seq_len: Optional[np.ndarray] = None,
        retrain_epochs: int = 50
    ):
        """
        Args:
            mu: Expected number of mutations to the full sequence (mu/L per position).
            recomb_rate: The probability of a crossover at any position in a sequence.
            threshold: At each round only sequences with fitness above
                (1-threshold)*f_max are retained as parents for generating next set of
                sequences.
            rho: The expected number of recombination partners for each recombinant.
            eval_batch_size: For code optimization; size of batches sent to model.

        """
        name = f"Adalead_mu={mu}_threshold={threshold}"
        print(name)

        self.threshold = threshold
        self.recomb_rate = recomb_rate
        self.alphabet = alphabet
        self.mu = mu  # number of mutations per *sequence*.
        self.rho = rho
        self.eval_batch_size = eval_batch_size
        self.model_args = model_args
        self.model = model
        self.sequences_batch_size = sequences_batch_size
        self.model_queries_per_batch = model_queries_per_batch
        self.model_contri = model_contri
        self.criterion_contri = criterion_contri
        self.optimizer_contri = optimizer_contri
        self.motif_size = motif_size
        self.motif_based = motif_based
        # Initial training data for retraining importance model
        self.initial_ohe = initial_ohe
        self.initial_y = initial_y
        self.initial_seq_len = initial_seq_len
        self.retrain_epochs = retrain_epochs

    def _recombine_population(self, gen):
        # If only one member of population, can't do any recombining
        if len(gen) == 1:
            return gen

        random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if random.random() < self.recomb_rate:
                    switch = not switch

                # putting together recombinants
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])

            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret

    def convert_to_ohe(self, data):
        N = len(data)
        ohe = np.zeros((N, self.model_args.max_len, \
            len(alphabets_of_color)))  ## batch size*sequenc_len*20
        seq_lengths = np.zeros((N,))

        for i, seq in enumerate(data):
            l = 0
            for j, aa in enumerate(seq):
                if aa == '0':  # padding character, keep as all zeros
                    continue
                ohe[i, j, aa_to_idx[aa]] = 1
                l += 1
            seq_lengths[i] = l

        return ohe, seq_lengths

    def calcualate_imp_aa(self,seq, seq_len, imp):
        num_seq = len(seq)
        max_len = int(max(seq_len))
        dis_seqs = np.zeros((num_seq, max_len), dtype=object)
        dis_imp = np.zeros((num_seq, max_len))
        for i in range(num_seq):
            l = int(seq_len[i])
            dis_seqs[i, 0:l] = list(seq[i])
            dis_imp[i, 0:l] = imp[i,0:l]

        dis_seqs = dis_seqs.reshape((-1,1))
        dis_imp = dis_imp.reshape((-1,1))
        aa_imp = [0]*len(self.alphabet)
        for j,alph in enumerate(self.alphabet):
            temp = (dis_seqs==alph)
            temp_imp = dis_imp[temp]
            aa_imp[j] = np.sum(temp_imp)/(np.sum(temp)+1E-18)

        return aa_imp

    def pad_seq(self, all_seq):
        list_of_lists = [list(map(str, group)) for group in all_seq]

        # Find the maximum length for padding
        max_length = max(len(lst) for lst in list_of_lists)

        # Pad lists with zeros
        padded_seq = np.array([lst + [0] * (max_length - len(lst)) for lst in list_of_lists])
        return padded_seq

    def normalize_impscore(self, imp_score):
        # Get the maximum value along axis=1 while keeping the dimensions for broadcasting
        max_vals = np.max(imp_score, axis=1, keepdims=True)
        # Normalize by dividing each element by the max value of its row
        imp_score = imp_score / (max_vals + 1E-18)
        return imp_score

    def make_motif_chunks(self):
        L = self.padded_seq.shape[1]  # Assuming L is the sequence length
        motif_dict = defaultdict(list)  # Dictionary to store motif chunks and their scores
        # Iterate through the sequence in steps of chunk_size
        for i in range(0, L, self.motif_size):
            chunk = self.padded_seq[:, i:i+self.motif_size]
            chunk_imp = self.normalized_score[:, i:i+self.motif_size]
            # Join along axis=1 (rows)
            joined = np.apply_along_axis(lambda x: ''.join(x), axis=1, arr=chunk)
            joined_imp = np.sum(chunk_imp, axis=1)

            # Store in dictionary (appending values)
            for motif, imp_score in zip(joined, joined_imp):
                motif_dict[motif].append(imp_score)

        # Compute the mean importance score for each motif
        motif_mean_dict = {motif: np.mean(scores) for motif, scores in motif_dict.items()}
        return motif_mean_dict

    def motif_level_assessment(self, roots, root_importance):
        ''' This function takes a bunch of sequences (roots)
        and mine all the motifs of size motif size
        and assign them importance score based on var
        root_importance '''
        self.padded_seq = self.pad_seq(roots)
        self.normalized_score = self.normalize_impscore(root_importance)
        return self.make_motif_chunks()

    def _retrain_importance_model(self, measured_sequences):
        """
        Retrain the importance model on initial data + all measured sequences.

        Args:
            measured_sequences: Dictionary containing "sequence" and "true_score" keys
        """
        # Convert measured sequences to one-hot encoding
        all_seqs = measured_sequences["sequence"]
        all_scores = np.array(measured_sequences["true_score"])

        # Convert sequences to OHE format
        measured_ohe, measured_seq_len = self.convert_to_ohe(all_seqs)

        # Combine with initial training data
        combined_ohe = np.concatenate([self.initial_ohe, measured_ohe], axis=0)
        combined_y = np.concatenate([self.initial_y.flatten(), all_scores.flatten()], axis=0)
        combined_seq_len = np.concatenate([self.initial_seq_len.flatten(), measured_seq_len.flatten()], axis=0)

        # Retrain the importance model
        print(f"Retraining importance model on {combined_ohe.shape[0]} samples for {self.retrain_epochs} epochs...")
        self.model_contri = train_importance_model(
            self.model_contri,
            self.criterion_contri,
            self.optimizer_contri,
            combined_ohe,
            combined_y,
            combined_seq_len,
            self.model_args.device,
            num_epochs=self.retrain_epochs
        )
        print("Importance model retraining complete.")

    def individual_protein_chunks(self, p, imp):
        '''p: individual protein sequence
        imp: imp score of each position
        returns: dict: {motifs_idx:importance}'''
        motif_dict = defaultdict(list)
        seq_length = len(p)
        # Slide over the sequence with a stride of 1
        for i in range(seq_length - self.motif_size + 1):
            motif = p[i:i+self.motif_size]  # Extract motif of size k
            motif_imp = np.mean(imp[i:i+self.motif_size])  # Sum importance scores for this motif
            motif_dict[i].append(motif_imp)  # Store importance scores

        # Compute mean importance for each unique motif
        protein_dict = {motif: np.mean(scores) for motif, scores in motif_dict.items()}
        return protein_dict



    def propose_sequences(
        self,measured_sequences: pd.DataFrame, is_imp_based= False, temp=1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""

        # Retrain importance model if we have initial training data and importance-based mutation is enabled
        if is_imp_based and self.initial_ohe is not None:
            self._retrain_importance_model(measured_sequences)

        measured_sequence_set = set(measured_sequences["sequence"])

        # Get all sequences within `self.threshold` percentile of the top_fitness
        top_fitness = max(measured_sequences["true_score"])
        top_inds = measured_sequences["true_score"] >= top_fitness * (
            1 - np.sign(top_fitness) * self.threshold
        )
        top_inds = top_inds.tolist()

        parents = np.resize(
            np.array(measured_sequences["sequence"])[top_inds],
            self.sequences_batch_size,
        )

        sequences = {}
        track_queries = 0
        while track_queries < self.model_queries_per_batch:
            # generate recombinant mutants
            for i in range(self.rho):
                parents = self._recombine_population(parents)

            for i in range(0, len(parents), self.eval_batch_size):
                # Here we do rollouts from each parent (root of rollout tree)
                roots = parents[i : i + self.eval_batch_size]
                root_ohe_fit, root_len_fit = self.convert_to_ohe(roots)
                root_fitnesses = predict_property(self.model_contri, root_ohe_fit, root_len_fit, self.model_args.device)
                root_ohe, root_len = self.convert_to_ohe(roots)
                if is_imp_based:
                    root_importance = contribution_score(self.model_contri, \
                        self.criterion_contri,self.optimizer_contri,root_ohe, root_len, \
                        self.model_args.device)

                    if not self.motif_based:
                        root_imp_aa = self.calcualate_imp_aa(roots, root_len, root_importance)
                        ## softmax #####
                        root_imp_aa = np.exp(np.array(root_imp_aa))
                        root_imp_aa = root_imp_aa/np.sum(root_imp_aa)
                        root_imp_aa = root_imp_aa.tolist()

                    elif self.motif_based:
                        '''Rank different motifs here'''
                        motif_mean_dict = self.motif_level_assessment(roots, root_importance)

                if self.rho > 0:
                    track_queries += len(root_fitnesses)

                nodes = list(enumerate(roots))
                while (
                    len(nodes) > 0
                    and track_queries
                    < self.model_queries_per_batch
                ):
                    child_idxs = []
                    children = []
                    while len(children) < len(nodes):
                        idx, node = nodes[len(children) - 1]
                        ## write a code here for importance
                        if not is_imp_based:
                            child = s_utils.generate_random_multiple_mutant(
                                node,
                                self.mu * 1 / len(node),
                                self.alphabet,
                                1
                            )
                        else:
                            node_imp = root_importance[idx,0:int(root_len[idx])]
                            if not self.motif_based:
                                ### importance-based mutation
                                child = s_utils.generate_importance_based_mutant(
                                    node,
                                    node_imp,
                                    temp,
                                    self.alphabet, root_imp_aa
                                )
                            elif self.motif_based:
                                node_motifs = self.individual_protein_chunks(node,node_imp)
                                ### importance-based motif level mutation
                                child = s_utils.motif_level_mutation(
                                    node,
                                    motif_mean_dict,
                                    node_motifs,
                                    self.motif_size, temp
                                )


                        # Stop when we generate new child that has never been seen
                        # before
                        if (
                            child not in measured_sequence_set
                            and child not in sequences
                        ):
                            child_idxs.append(idx)
                            children.append(child)


                    children_ohe, children_len = self.convert_to_ohe(children)
                    fitnesses = predict_property(self.model_contri, children_ohe, children_len, self.model_args.device)
                    track_queries += len(fitnesses)
                    sequences.update(zip(children, fitnesses))

                    nodes = []
                    for idx, child, fitness in zip(child_idxs, children, fitnesses):
                        if fitness >= root_fitnesses[idx]:
                            nodes.append((idx, child))

        if len(sequences) == 0:
            raise ValueError(
                "No sequences generated. If `model_queries_per_batch` is small, try "
                "making `eval_batch_size` smaller"
            )

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)

        return new_seqs[sorted_order], preds[sorted_order]
