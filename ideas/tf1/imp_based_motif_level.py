from adalead_akash import Adalead
import sys
import numpy as np
import argparse
import time
# sys.path.append("/home/apa2237/generative_model_work/oracle_models/amp")
sys.path.append("/home/apa2237/generative_model_work/datasets/tf_ARX_L343Q_R1_8mers/")
sys.path.append("/home/apa2237/generative_model_work/importance_models/tf_ARX_L343Q_R1_8mers/")
# from produce_output import initalize,test
from generate_property import output_property_oracle
from generate_property import onehotseq
from contributions_score_generative import initalize as initalize_contri
import json


import json

import pickle
# parser = argparse.ArgumentParser()

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Run Adalead with specified temperature.')
parser.add_argument('--temp', type=float, default=1, help='Temperature parameter for Adalead')
parser.add_argument('--device', type=str, default='cuda')
input_args = parser.parse_args()

print(f'========= Temperature = {input_args.temp} ===========')

alphabet = np.load('/home/apa2237/generative_model_work/datasets/tf_ARX_L343Q_R1_8mers/categorical_variables.npy', allow_pickle=True).tolist()
data_path = "/home/apa2237/generative_model_work/datasets/tf_ARX_L343Q_R1_8mers/"
output_property_oracle = output_property_oracle()

top_per = 30

data_path = "/home/apa2237/generative_model_work/datasets/tf_ARX_L343Q_R1_8mers/"

seq_start = np.load(f'{data_path}/seq_test.npy', allow_pickle=True).reshape((-1,1))
y_start = np.load(f'{data_path}/y_test.npy', allow_pickle=True).reshape((-1,1))

print('Before filtering',len(seq_start), len(y_start))
print('Mean before', np.mean(y_start))
print('Min before', np.min(y_start))
print('Max before', np.max(y_start))

cap = max(y_start)
floor = min(y_start)
cutoff = 0.2
print('Cutoff', cutoff)
below_idx = (y_start<cutoff)
# print('Below_idx', np.sum(below_idx*1))
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
    max_len = 8
model_args = args()
# model = initalize(model_args)

# print(model)
print(model_args.device)

seq_start = seq_start.tolist()
start_fitness = output_property_oracle.output_property(seq_start).tolist()
# start_fitness = test(model,seq_start,model_args).tolist()


start_num_samples = len(seq_start)

### loading contribution score model
model_contri, criterion_contri,optimizer_contri = initalize_contri(model_args.device)

# all_ng = [3,5,10,20,50,100,200,500]
all_ng = [20,50,100]
# all_ng = [3]
num_trials = 10 # number of trials with one ng value for reproducibility
rounds = 10 ## number of generation iteration in each trials

adalead_imp_code_book = {}
adalead_new_sequences = {}

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
    optimizer_contri =  optimizer_contri,
    motif_size = 1,
    motif_based=True)
    
    is_imp_based = True
    temp = input_args.temp ## temp value


    small_code_book = np.zeros((num_trials, rounds+1, 3))
    
    new_sequences = {trial: {} for trial in range(num_trials)}
    
    for trial in range(num_trials):
        starting_mean = np.mean(y_start[0:ng])
        set_seeds(trial)
        print(f'For Num_samples:{ng}, Iteration: {trial+1}')
        # print(np.mean(start_fitness))
        bundle = {"sequence":seq_start[0:start_num_samples], "true_score":start_fitness[0:start_num_samples]}
        m, maxx = np.mean(np.array(bundle["true_score"])), max(np.array(bundle["true_score"]))
        small_code_book[trial, 0, :] = [m,maxx,starting_mean]
        print(f'At the beginning of starting, Mean:{m}, Max:{maxx} ')
        for i in range(1,rounds+1):
            new_seq, new_prop = adalead_algo.propose_sequences(bundle,\
                is_imp_based, temp)
            bundle["sequence"].extend(new_seq.tolist())
            bundle["true_score"].extend(new_prop.tolist())
            m, maxx, new_mean = np.mean(np.array(bundle["true_score"])), \
                max(np.array(bundle["true_score"])), \
                np.mean(new_prop)

            # store the actual new sequences for this trial & round
            new_sequences[trial][i] = new_seq.tolist()

            small_code_book[trial,i,:] = [m,maxx,new_mean]
            print(f'Iteration {i}, Mean:{m}, Max:{maxx}, New Samples: {new_mean} ')   

        # print(new_seq)
            
    adalead_imp_code_book[ng] = small_code_book
    adalead_new_sequences[ng] = new_sequences  # <-- add this
    # print(aaaa)

    np.save(f'./generative_results/our_{input_args.temp}', adalead_imp_code_book)
    np.save(f'./generative_results/sequences_our_{input_args.temp}', adalead_new_sequences)