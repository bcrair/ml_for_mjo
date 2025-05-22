#!/usr/bin/env python
# coding: utf-8

# ## ViT for MJO Index Prediciton - MDL Train 
# ## Normal Learning (Part 1)
# #### Ben Crair
# #### May 6th, 2025


deBug = True


# ### Section A: Parameters

tot = 43076
lat = 30
lon= 180
var = 5
seed_num = 1

# Parameters are set based on the reference code.

#imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from netCDF4 import Dataset
from vit_pytorch import ViT
import random

# setup random seeding
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

torch.cuda.is_available()

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

# ### Section B: Read in Data

# read data: (time,lat,lon) and (time)
leadTm = 15 # [SXM-ensemble-change]
directory = "/home/bec32/palmer_scratch/ML.MJO/Data/"
in1  = Dataset(directory + "CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_TMQ_leadTm"   +str(leadTm)+".nc") # prcpW
in2  = Dataset(directory + "CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_FLUT_leadTm"  +str(leadTm)+".nc") # OLR
in3  = Dataset(directory + "CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_U200_leadTm"  +str(leadTm)+".nc") # U200
in4  = Dataset(directory + "CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_U850_leadTm"  +str(leadTm)+".nc") # U850
in5  = Dataset(directory + "CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_TREFHT_leadTm"+str(leadTm)+".nc") # T2
out1 = Dataset(directory + "CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC1.nc")                                      # RMM1
out2 = Dataset(directory + "CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC2.nc")                                      # RMM2
if deBug:
  print("in1  = ", in1,  "\n")
  print("out1 = ", out1, "\n")

# aggregate data: (time,lat,lon,var) and (time, rmms)
data   = np.zeros((tot,lat,lon,var)).astype(np.float32) # (time,lat,lon,var)
target = np.zeros((tot,2)).astype(np.float32)           # (time,rmms)
data[...,0] = in1.variables["TMQ"]   [:,:,:]            # take all data and use padding (dfrnt from Shin et al. 2024 that mod%2=0 for the first three convolutional layers)
data[...,1] = in2.variables["FLUT"]  [:,:,:]            # take all data and use padding (dfrnt from Shin et al. 2024 that mod%2=0 for the first three convolutional layers)
data[...,2] = in3.variables["U200"]  [:,:,:]            # take all data and use padding (dfrnt from Shin et al. 2024 that mod%2=0 for the first three convolutional layers)
data[...,3] = in4.variables["U850"]  [:,:,:]            # take all data and use padding,(dfrnt from Shin et al. 2024 that mod%2=0 for the first three convolutional layers)
data[...,4] = in5.variables["TREFHT"][:,:,:]            # take all data and use padding,(dfrnt from Shin et al. 2024 that mod%2=0 for the first three convolutional layers)
target[:,0] = out1.variables["PC1"]  [:]
target[:,1] = out2.variables["PC2"]  [:]
if deBug:
  print("data.shape   = ", data.shape,   "\n")
  print("target.shape = ", target.shape, "\n")

# The above code is derived from reference.


# convert data into order expected by Pytorch
data = data.transpose(0,3,1,2)
print(data.shape)

# Define dataset class
class MDLDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features).to(device)
        self.targets = torch.tensor(targets).to(device)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        return feature, target

dataset = MDLDataset(data, target)

mdl_total_samples = len(dataset)
mdl_train_size = int(0.8 * mdl_total_samples) - int(0.8 * mdl_total_samples) % 121
mdl_test_size = mdl_total_samples - mdl_train_size

mdl_train_dataset, mdl_test_dataset = torch.utils.data.random_split(dataset, [mdl_train_size, mdl_test_size])

batch_size = 121

train_dataloader = torch.utils.data.DataLoader(mdl_train_dataset, batch_size=batch_size, shuffle = True, pin_memory=False)
test_dataloader = torch.utils.data.DataLoader(mdl_test_dataset, batch_size=batch_size, shuffle = False, pin_memory=False)


# ### Section C: ViT Architecture

model = ViT(
    #check: what to do about rectangular image size
    #set image size as maximum of width and height for rectangular images
    image_size = 180,
    #we use patch size of 6*6
    patch_size = 15,
    #we use two output "classes" as out RMM values
    num_classes = 2,
    #I beleive this is the embedding dimension
    #technically according to documentation the "Last dimension of output tensor after linear transformation nn.Linear(..., dim)"
    dim = 256,
    #start with 6 layers (transfomer blocks)
    depth=6,
    #number of heads in multi-head attention layer
    heads=8,
    #dimesion of the MLP layer
    mlp_dim = 512,
    #number of image channels
    channels = 5,
    #dropout rate
    dropout=0.1,
    #embedding dropout trade
    emb_dropout=0.1
)


# Hyperparameters
#maybe tre 0.001
lr = 0.0005
epochs = 500


# ### Section D: Compile

batches_per_datset = len(train_dataloader)
print(f"batches per dataset: {batches_per_datset}")

train_losses = []
test_losses = []

# send to GPU, if available
model = model.to(device)
# Optimizer, note that we use weight decay with coefficinet 1e-5
# Note that here we choose AdamW optimizer over Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
# Reduce Learning Rate on Plateau Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001, factor = 0.5, mode='min')
loss_fn = torch.nn.MSELoss()

# define early stopping class
class EarlyStopping:
    def __init__(self, threshold, patience):
        self.threshold = threshold
        self.patience = patience
        self.patience_count = 0
        
    def check_early_stop(self, test_loss_history):
        #expect last item of test_loss_history to be most recent loss
        if len(test_loss_history) < self.patience:
            return False
        
        if min(test_loss_history[:-1]) - test_loss_history[-1] < self.threshold:
            self.patience_count += 1
            if self.patience_count >= self.patience:
                return True
        else:
            self.patience_count = 0
            return False
        
earlystopping = EarlyStopping(threshold = 0.0001, patience = 10)
        
print(model)


# ### Section E: Train

with torch.no_grad():
    model.eval()
    total_train_loss = 0.0
    for batch_data, batch_target in train_dataloader:

        y = model(batch_data)
        # Loss
        loss = loss_fn(y, batch_target)
        total_train_loss += loss.item() * batch_data.size(0)

    avg_train_loss = total_train_loss / len(train_dataloader.dataset)

    total_test_loss = 0.0
    for batch_data, batch_target in test_dataloader:

        pred_y = model(batch_data)	
        loss = loss_fn(pred_y, batch_target)
        total_test_loss += loss.item() * batch_data.size(0)

    avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    print(f"Initial Performance of Untrained Model: \nAvg Train Loss: {avg_train_loss}, Avg Test Loss: {avg_test_loss}")
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

for i in range(epochs):
    total_train_loss = 0.0
    for batch_data, batch_target in train_dataloader:

        model.train()

        y = model(batch_data)
        # Loss
       
        loss = loss_fn(y, batch_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * batch_data.size(0)
    avg_train_loss = total_train_loss / len(train_dataloader.dataset)

    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch_data, batch_target in test_dataloader:

            pred_y = model(batch_data)	
            loss = loss_fn(pred_y, batch_target)
            total_test_loss += loss.item() * batch_data.size(0)

    avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

    #step scheduler
    scheduler.step(avg_test_loss)
    print(f"Epoch: {i + 1}, Avg Train Loss: {avg_train_loss}, Avg Test Loss: {avg_test_loss}, LR: {scheduler.get_last_lr()}")
        
    if earlystopping.check_early_stop(test_losses):
        break
    
torch.save(model, "/home/bec32/project/ML_for_MJO/ViT/Ensemble_Results/Result_01/ViTMJO_Ben_MDL_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+".pth")

print(f"\nCompleted: {__file__}")



