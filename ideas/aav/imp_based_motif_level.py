from main import Adalead
import sys
import os
import numpy as np
import argparse
import time

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the IDEAS root directory (two levels up from ideas/aav/)
IDEAS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Add paths relative to IDEAS root
sys.path.append(os.path.join(IDEAS_ROOT, 'datasets', 'aav'))
sys.path.append(os.path.join(IDEAS_ROOT, 'importance_models', 'aav'))

from contributions_score_generative import initalize as initalize_contri, predict_property
import json
import pickle

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Run Adalead with specified temperature for AAV fitness optimization.')
parser.add_argument('--temp', type=float, default=1, help='Temperature parameter for Adalead')
parser.add_argument('--device', type=str, default='cuda')
input_args = parser.parse_args()

print(f'========= AAV Fitness Optimization - Temperature = {input_args.temp} ===========')

# Data path relative to IDEAS root
data_path = os.path.join(IDEAS_ROOT, 'datasets', 'aav')

alphabet = np.load(os.path.join(data_path, 'categorical_variables.npy'), allow_pickle=True).tolist()
# Pre-compute alphabet index mapping for O(1) lookup
aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}

seq_start = np.load(os.path.join(data_path, 'seq_test.npy'), allow_pickle=True).reshape((-1,1))
y_start = np.load(os.path.join(data_path, 'y_test.npy'), allow_pickle=True).reshape((-1,1))

print('Before filtering',len(seq_start), len(y_start))
print('Mean before', np.mean(y_start))
print('Min before', np.min(y_start))
print('Max before', np.max(y_start))

cutoff = 0.2
print('Cutoff', cutoff)
below_idx = (y_start<cutoff)
print(below_idx.shape,seq_start.shape)

seq_start = seq_start[below_idx]
y_start = y_start[below_idx]

print('Filtering==================')
print('After filtering',len(seq_start), len(y_start))
print('Mean after', np.mean(y_start))
print('Max after', np.max(y_start))

sorted_idx = np.argsort(y_start)[::-1]
seq_start = seq_start[sorted_idx]
print(seq_start.shape)
y_start = y_start[sorted_idx]


class args():
    device = input_args.device
    max_len = 750
model_args = args()

print(model_args.device)

### loading contribution score model (used for both oracle and importance)
model_contri, criterion_contri, optimizer_contri = initalize_contri(model_args.device)

def convert_to_ohe(data, max_len):
    N = len(data)
    ohe = np.zeros((N, max_len, len(alphabet)))
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

seq_start = seq_start.tolist()
start_ohe, start_len = convert_to_ohe(seq_start, model_args.max_len)
start_fitness = predict_property(model_contri, start_ohe, start_len, model_args.device).tolist()

start_num_samples = len(seq_start)

### loading initial training data for importance model retraining
initial_ohe = np.load(os.path.join(data_path, 'x_train.npy'), allow_pickle=True)
initial_y = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True)
initial_seq_len = np.load(os.path.join(data_path, 'len_train.npy'), allow_pickle=True)
print(f'Loaded initial training data: {initial_ohe.shape[0]} samples for importance model retraining')

# all_ng = [3,5,10,20,50,100,200,500]
all_ng = [20,50,100]
num_trials = 10 # number of trials with one ng value for reproducibility
rounds = 10 ## number of generation iteration in each trials

adalead_imp_code_book = {}
adalead_new_sequences = {}
adalead_iteration_times = {}

def set_seeds(trial):
    seed = trial
    print(f"Seed: {seed}")
    np.random.seed(seed)

print('============= Are you happy with the choice of motif size ===========')
time.sleep(1.0)

for count_ng,ng in enumerate(all_ng):
    adalead_algo = Adalead(None,
    model_args,
    rounds = 1,
    sequences_batch_size = ng, #keeping sequences_batch_size == model_queries_per_batch
    model_queries_per_batch = ng,
    starting_sequence = None,
    alphabet = alphabet,
    mu = 1,
    recomb_rate = 0.2,
    threshold =  0.05,
    rho = 0,
    eval_batch_size = 20,
    model_contri = model_contri,
    criterion_contri = criterion_contri,
    optimizer_contri = optimizer_contri,
    motif_size = 15,
    motif_based=True,
    initial_ohe = initial_ohe,
    initial_y = initial_y,
    initial_seq_len = initial_seq_len,
    retrain_epochs = 50)

    is_imp_based = True
    temp = input_args.temp ## temp value


    small_code_book = np.zeros((num_trials, rounds+1, 3))
    iteration_times = np.zeros((num_trials, rounds))

    new_sequences = {trial: {} for trial in range(num_trials)}

    for trial in range(num_trials):
        starting_mean = np.mean(y_start[0:ng])
        set_seeds(trial)
        print(f'For Num_samples:{ng}, Iteration: {trial+1}')
        bundle = {"sequence":seq_start[0:start_num_samples], "true_score":start_fitness[0:start_num_samples]}
        m, maxx = np.mean(np.array(bundle["true_score"])), max(np.array(bundle["true_score"]))
        small_code_book[trial, 0, :] = [m,maxx,starting_mean]
        print(f'At the beginning of starting, Mean:{m}, Max:{maxx} ')
        for i in range(1,rounds+1):
            iter_start_time = time.time()
            new_seq, new_prop = adalead_algo.propose_sequences(bundle,\
                is_imp_based, temp)
            iter_end_time = time.time()
            iter_elapsed = iter_end_time - iter_start_time
            iteration_times[trial, i-1] = iter_elapsed

            bundle["sequence"].extend(new_seq.tolist())
            bundle["true_score"].extend(new_prop.tolist())
            m, maxx, new_mean = np.mean(np.array(bundle["true_score"])), \
                max(np.array(bundle["true_score"])), \
                np.mean(new_prop)

            # store the actual new sequences for this trial & round
            new_sequences[trial][i] = new_seq.tolist()

            small_code_book[trial,i,:] = [m,maxx,new_mean]
            print(f'Iteration {i}, Mean:{m}, Max:{maxx}, New Samples: {new_mean}, Time: {iter_elapsed:.4f}s')

    adalead_imp_code_book[ng] = small_code_book
    adalead_new_sequences[ng] = new_sequences
    adalead_iteration_times[ng] = iteration_times

    results_dir = os.path.join(SCRIPT_DIR, 'generative_results')
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f'our_{input_args.temp}'), adalead_imp_code_book)
    np.save(os.path.join(results_dir, f'sequences_our_{input_args.temp}'), adalead_new_sequences)
    np.save(os.path.join(results_dir, f'our_iteration_times_{input_args.temp}'), adalead_iteration_times)
