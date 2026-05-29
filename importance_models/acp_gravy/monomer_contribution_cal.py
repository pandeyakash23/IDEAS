import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import matplotlib as mpl
import os
import gc
import pandas as pd
import csv
from numpy import *
from torch.utils.tensorboard import SummaryWriter
from datetime import date
import time
import builtins
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, mean_absolute_error, r2_score, mean_squared_error
from complor import dataset, complor_network

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the IDEAS root directory (two levels up from importance_models/acp_gravy/)
IDEAS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# Model directory relative to script
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
# Data path relative to IDEAS root
DATA_PATH = os.path.join(IDEAS_ROOT, 'datasets', 'acp_gravy')

which_data = input('Enter the dataset for which you want to calculate the token importance (train,valid,test):')

## Dataloader
batch_size = 256
class spiderdataset(Dataset) :
    def __init__(self,ohe,seq_len,n_samples) :
        # data loading
        self.ohe = torch.from_numpy(ohe.astype(np.float32))
        self.seq_len = torch.from_numpy(seq_len.astype(int64))
        self.n_samples = n_samples 
        
    def __getitem__(self,index) :
        return self.ohe[index], self.seq_len[index]
    def __len__(self):    
        return self.n_samples      

def make_dataset(ohe_valid, seq_len_valid): 

 
    test_dataset = spiderdataset(ohe_valid,seq_len_valid,ohe_valid.shape[0])

      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return  test_loader, ohe_valid.shape[0], ohe_valid.shape[1]

    
def initalize():
    device = 'cuda:2'
    color_args = np.load(os.path.join(MODEL_DIR, 'save_dict.npy'), allow_pickle=True).tolist()
    model = complor_network(1, color_args['q'], color_args['d'], color_args['max_m'], device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best.pth')))
    rank = device
    model.eval().to(rank)
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    return model, criterion, optimizer

def motif_identification(ohe, seq_len):  
    test_loader, test_size, max_seq_len  = make_dataset(ohe, seq_len)
    model, criterion, optimizer = initalize()
    rank = next(model.parameters()).device 
    store_importance = torch.zeros((test_size, max_seq_len)).to(rank)
    count_test = 0
    for _, (i_x,i_seq) in enumerate(test_loader):
        i_x = i_x.to(rank) #.type(dtype=torch.float32)
        i_seq = i_seq.to(rank).type(dtype=torch.float32)
        i_batch = i_x.size(0)
        iter_y_pred, cam,initial_cam,initial_pool = model.forward_motif_importance(i_x, i_seq)

        base_loss = torch.sum(iter_y_pred)
        optimizer.zero_grad()
        base_loss.backward()
        
        # cam = torch.abs(cam[0])
        cam = cam[0]
        print('Size of the gradient',cam.size())

        
        for prot in range(cam.size(0)):
            cam[prot,...] = cam[prot,...]/(torch.max(abs(cam[prot,...]))+1E-18)
            
        for m_i in range(cam.size(-1)):
            mo_level_imp = \
                model.calculate_motif_level(cam[...,m_i], m_i+1)
            

            kernel_size = max_seq_len - mo_level_imp.size(-1) + 1
            store_importance[count_test:count_test+i_batch,...] += model.assigning_importance(mo_level_imp, kernel_size, max_seq_len)

        
        count_test += i_batch
    
    with torch.no_grad():
        store_importance = store_importance.to('cpu').numpy()
        print(store_importance[15])
        np.save(os.path.join(MODEL_DIR, f'importance_{which_data}'), store_importance)

    # return store_importance


if __name__ == '__main__':
    cp_1 = time.time()
    ohe_send = np.load(os.path.join(DATA_PATH, f'x_{which_data}.npy'), allow_pickle=True)
    seq_send = np.load(os.path.join(DATA_PATH, f'len_{which_data}.npy'), allow_pickle=True)
    motif_identification(ohe_send, seq_send)
    cp_2 = time.time()
    print('Time Taken', cp_2 - cp_1)
    
    
