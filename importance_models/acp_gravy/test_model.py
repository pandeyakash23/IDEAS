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
from sklearn.metrics import r2_score, mean_absolute_error
from complor import dataset, complor_network

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the IDEAS root directory (two levels up from importance_models/acp_gravy/)
IDEAS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# Model directory relative to script
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
# Data path relative to IDEAS root
DATA_PATH = os.path.join(IDEAS_ROOT, 'datasets', 'acp_gravy')

## Dataloader
batch_size = 256


def make_dataset():
    ohe_valid = np.load(os.path.join(DATA_PATH, 'x_test.npy'), allow_pickle=True)
    classes_valid = np.argmax(ohe_valid, axis=2)
    output_valid = np.load(os.path.join(DATA_PATH, 'y_test.npy'), allow_pickle=True)
    seq_len_valid = np.load(os.path.join(DATA_PATH, 'len_test.npy'), allow_pickle=True) 
  
 
    test_dataset = dataset(ohe_valid,classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])

      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return  test_loader, ohe_valid.shape[0]

    
def initalize():
    device = 'cuda:2'
    color_args = np.load(os.path.join(MODEL_DIR, 'save_dict.npy'), allow_pickle=True).tolist()
    model = complor_network(1, color_args['q'], color_args['d'], color_args['max_m'], device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best.pth')))
    rank = device
    model.eval().to(rank)
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()

    return model, criterion

def test():
    test_loader, valid_size  = make_dataset()
    model, criterion = initalize()
    rank = next(model.parameters()).device 
    with torch.no_grad():
        predicted_label = torch.zeros((valid_size, 1))
        actual_label = torch.zeros((valid_size, 1))
        count_valid = 0         
        valid_loss = 0
        for j, (i_x,i_classes, i_seq, i_actual) in enumerate(test_loader):
            i_x = i_x.to(rank) #.type(dtype=torch.float32)
            i_seq = i_seq.to(rank).type(dtype=torch.float32)
            i_classes = i_classes.to(rank)
            i_actual = i_actual.to(rank)
            
            # forward pass    
            iter_y_pred = model(i_x, i_seq)
            loss = criterion(iter_y_pred, i_actual)
            valid_loss = (valid_loss*j + loss.item())/(j+1)
            size = iter_y_pred.size(0)
            predicted_label[count_valid:count_valid+size, :] = iter_y_pred 
            actual_label[count_valid:count_valid+size, :] = i_actual
            count_valid += size
            
            predicted_label = predicted_label.cpu().numpy().reshape((-1,1))
            actual_label = actual_label.cpu().numpy().reshape((-1,1))
            valid_r2 = r2_score(actual_label, predicted_label)
            valid_mae = mean_absolute_error(actual_label, predicted_label)
            
        print(f'Test acc:{valid_r2}, MAE: {valid_mae}')

        
if __name__=='__main__':
    cp_1 = time.time()
    test()
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
