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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,mean_absolute_error,r2_score, mean_squared_error
from complor import dataset, complor_network

# Training dataset class for retraining
class TrainDataset(Dataset):
    def __init__(self, ohe, classes, seq_len, output, n_samples):
        self.ohe = torch.from_numpy(ohe.astype(np.float32))
        self.seq_len = torch.from_numpy(seq_len.astype(np.int64))
        self.classes = torch.from_numpy(classes.astype(np.int64))
        self.output = torch.from_numpy(output.astype(np.float32)).reshape((n_samples, 1))
        self.n_samples = n_samples

    def __getitem__(self, index):
        return self.ohe[index], self.classes[index], self.seq_len[index], self.output[index]

    def __len__(self):
        return self.n_samples

   
# which_data = input('Enter the dataset for which you want to calculate the token importance (train,valid,test):')

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

    
def initalize(rank):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'model')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    color_args = np.load(os.path.join(model_dir, 'save_dict.npy'), allow_pickle=True).tolist()
    model = complor_network(1, color_args['q'], color_args['d'], color_args['max_m'], rank)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best.pth')))
    model.eval().to(rank)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    return model, criterion, optimizer


def train_importance_model(model, criterion, optimizer, ohe_combined, y_combined, seq_len_combined,
                           rank, num_epochs=50, batch_size=256):
    """
    Retrain the importance model on combined data (original + newly generated).

    Args:
        model: The importance model to train
        criterion: Loss function (MSE)
        optimizer: Optimizer (Adam)
        ohe_combined: One-hot encoded sequences (numpy array)
        y_combined: Target values (numpy array)
        seq_len_combined: Sequence lengths (numpy array)
        rank: Device (cuda/cpu)
        num_epochs: Number of training epochs (default 50)
        batch_size: Batch size for training (default 256)

    Returns:
        model: The trained model
    """
    # Put model in training mode
    model.train()

    # Create classes from one-hot encoding
    classes_combined = np.argmax(ohe_combined, axis=2)

    # Create dataset and dataloader
    n_samples = ohe_combined.shape[0]
    train_dataset = TrainDataset(ohe_combined, classes_combined, seq_len_combined, y_combined, n_samples)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        avg_loss = 0
        for i, (i_x, i_classes, i_seq, i_actual) in enumerate(train_loader):
            i_x = i_x.to(rank)
            i_seq = i_seq.to(rank).type(dtype=torch.float32)
            i_classes = i_classes.to(rank)
            i_actual = i_actual.to(rank)

            # Forward pass
            iter_y_pred = model(i_x, i_seq)
            loss = criterion(iter_y_pred, i_actual)
            avg_loss = (avg_loss * i + loss.item()) / (i + 1)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Set model back to eval mode for inference
    model.eval()

    return model

def contribution_score(model, criterion,optimizer,ohe, seq_len,rank):  
    test_loader, test_size, max_seq_len  = make_dataset(ohe, seq_len)
    # model, criterion, optimizer = initalize(rank)
    # rank = next(model.parameters()).device 
    store_importance = torch.zeros((test_size, max_seq_len)).to(rank)
    count_test = 0
    for _, (i_x,i_seq) in enumerate(test_loader):
        i_x = i_x.to(rank) #.type(dtype=torch.float32)
        i_seq = i_seq.to(rank).type(dtype=torch.float32)
        i_batch = i_x.size(0)
        iter_y_pred, cam,_,_ = model.forward_motif_importance(i_x, i_seq)

        base_loss = torch.sum(iter_y_pred)
        optimizer.zero_grad()
        base_loss.backward()
        

        cam = cam[0]
        # print('Size of the gradient',cam.size())

        
        for prot in range(cam.size(0)):
            cam[prot,...] = cam[prot,...]/(torch.max(abs(cam[prot,...]))+1E-18)
            
        for m_i in range(cam.size(-1)):
            mo_level_imp = \
                model.calculate_motif_level(cam[...,m_i], m_i+1)
            

            kernel_size = max_seq_len - mo_level_imp.size(-1) + 1
            store_importance[count_test:count_test+i_batch,...] += model.assigning_importance(mo_level_imp, kernel_size, max_seq_len)

        
        count_test += i_batch
    
    with torch.no_grad():   
        return store_importance.to('cpu').numpy()
        # print(store_importance[15])
        # np.save(f'./model/importance_{which_data}', store_importance)
    
    # return store_importance
    

        
# if __name__=='__main__':
#     path =  '/home/apa2237/Data_constrained_rep_learning/toy_problems/antimicrobial_classification/'        
#     cp_1 = time.time()
#     ohe_send = np.load(f'{path}/x_{which_data}.npy', allow_pickle=True)
#     seq_send = np.load(f'{path}/len_{which_data}.npy', allow_pickle=True)
#     motif_identification(ohe_send, seq_send)
#     cp_2 = time.time()
#     print('Time Taken',cp_2-cp_1)
    
    
