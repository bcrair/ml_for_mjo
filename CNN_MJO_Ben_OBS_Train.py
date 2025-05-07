# ## CNN for MJO Index Prediciton 
# ## Transfer Learning (Step 2)
# #### Ben Crair
# #### May 6th, 20225

deBug = True

# ### Section A: Parameters

# 1. pct
# for tot=5092
#   1) pct = 1528, then pct/tot = 1528/5092 = 0.300079 ~= 0.3
#   2) pct = 2036, then pct/tot = 2036/5092 = 0.399843 ~= 0.4
#   3) 1528+1528+2036 = 5092, corresponding to the cross-validation 3:3:4 ratio (Shin et al. 2024, P6)
tot = 5092 # 42-year DJFM out of 1979-2018 (leap years considered)
pct = 1528 # 1528 or 2036 for cross-validation (Shin2024ORGN: "pct = 1528 # 2036")
lat = 30   # num of lat points
lon = 180  # num of lon points
var = 5    # num of input vars (channels in CNN)

seed_num = 1

# Above is from collaborators

#imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from netCDF4 import Dataset
import os
import random
import copy

np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

print(f"GPU Available: {torch.cuda.is_available()}")

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

# ### Load in Data for Transfer Learning (Section B)

# read data: (time,lat,lon) and (time)
leadTm = 15 # [SXM-ensemble-change]
directory = "/home/bec32/palmer_scratch/ML.MJO/Data/"

in1  = Dataset(directory + "CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_tcwv_leadTm"  +str(leadTm)+".nc") # prcpW
in2  = Dataset(directory + "CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_olr_leadTm"   +str(leadTm)+".nc") # OLR
in3  = Dataset(directory + "CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_u200_leadTm"  +str(leadTm)+".nc") # U200
in4  = Dataset(directory + "CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_u850_leadTm"  +str(leadTm)+".nc") # U850
in5  = Dataset(directory + "CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_trefht_leadTm"+str(leadTm)+".nc") # T2
out1 = Dataset(directory + "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC1.nc")                                      # RMM1
out2 = Dataset(directory + "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC2.nc")                                      # RMM2
if deBug:
  print("in1  = ", in1,  "\n")
  print("out1 = ", out1, "\n")

# aggregate data: (time,lat,lon,var) and (time, rmms)
data   = np.zeros((tot,lat,lon,var)).astype(np.float32) # (time,lat,lon,var)
target = np.zeros((tot,2)).astype(np.float32)           # (time,rmms)
data[...,0] = in1.variables["tcwv"]  [:,:,:]            # take all data and use padding (dfrnt from Shin et al. 2024 that mod%2=0 for the first three convolutional layers)
data[...,1] = in2.variables["olr"]   [:,:,:]            # take all data and use padding (dfrnt from Shin et al. 2024 that mod%2=0 for the first three convolutional layers)
data[...,2] = in3.variables["u200"]  [:,:,:]            # take all data and use padding (dfrnt from Shin et al. 2024 that mod%2=0 for the first three convolutional layers)
data[...,3] = in4.variables["u850"]  [:,:,:]            # take all data and use padding (dfrnt from Shin et al. 2024 that mod%2=0 for the first three convolutional layers)
data[...,4] = in5.variables["trefht"][:,:,:]            # take all data and use padding (dfrnt from Shin et al. 2024 that mod%2=0 for the first three convolutional layers)
target[:,0] = out1.variables["PC1"]  [:]
target[:,1] = out2.variables["PC2"]  [:]
if deBug:
  print("data.shape   = ", data.shape,   "\n")
  print("target.shape = ", target.shape, "\n")

# Above is data loading code from collborators.

# convert data into order expected by Pytorch
data = data.transpose(0,3,1,2)
print(data.shape)

# define batch size
batch_size = 121

# Define dataset class

class OBSDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features).to(device)
        self.targets = torch.tensor(targets).to(device)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        return feature, target
    
test_ts = []
val_ts = []

# for round 1
j = 1
pct = 1528
# training dataset (formulate the training dataset by cutting off the original data and the cutoff[j*pct:(j+1)*pct] becomes the test dataset)
train_x_1 = np.delete(data,   np.s_[j*pct:(j+1)*pct], axis=0)
train_y_1 = np.delete(target, np.s_[j*pct:(j+1)*pct], axis=0)

# test datasets (formulate the test dataset by taking the cutoff[j*pct:(j+1)*pct] when formulating the training dataset)
test_x_1  = data  [j*pct:(j+1)*pct]
test_y_1  = target[j*pct:(j+1)*pct]

test_t_1  = np.shape(test_y_1)[0] # time-dim of the test dataset

train_data_1 = OBSDataset(train_x_1, train_y_1)

N = len(train_data_1)
train_len = int(N * 0.8)

train_dataset_1 = torch.utils.data.Subset(train_data_1, list(range(0, train_len)))
val_dataset_1 = torch.utils.data.Subset(train_data_1, list(range(train_len, N)))
test_dataset_1 = OBSDataset(test_x_1, test_y_1)

train_dataloader1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=batch_size, shuffle = True, pin_memory=False)
val_dataloader1 = torch.utils.data.DataLoader(val_dataset_1, batch_size=batch_size, shuffle = True, pin_memory=False)

test_dataloader1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=batch_size, shuffle = False, pin_memory=False)
test_ts.append(test_t_1)
val_ts.append(len(val_dataloader1.dataset))

if deBug:
  print("train_x_1.shape = ", train_x_1.shape)
  print("train_y_1.shape = ", train_y_1.shape)
  print("test_x_1.shape  = ", test_x_1.shape)
  print("test_y_1.shape  = ", test_y_1.shape)
  print("test_t_1        = ", test_t_1)

# for round 2
j = 2
pct = 1528
# training dataset (formulate the training dataset by cutting off the original data and the cutoff[j*pct:(j+1)*pct] becomes the test dataset)
train_x_2 = np.delete(data,   np.s_[j*pct:], axis=0)
train_y_2 = np.delete(target, np.s_[j*pct:], axis=0)

# test datasets (formulate the test dataset by taking the cutoff[j*pct:(j+1)*pct] when formulating the training dataset)
test_x_2  = data  [j*pct:]
test_y_2  = target[j*pct:]

test_t_2  = np.shape(test_y_2)[0] # time-dim of the test dataset

train_data_2 = OBSDataset(train_x_2, train_y_2)

N = len(train_data_2)
train_len = int(N * 0.8)

train_dataset_2 = torch.utils.data.Subset(train_data_2, list(range(0, train_len)))
val_dataset_2 = torch.utils.data.Subset(train_data_2, list(range(train_len, N)))
test_dataset_2 = OBSDataset(test_x_2, test_y_2)

train_dataloader2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size, shuffle = True, pin_memory=False)
val_dataloader2 = torch.utils.data.DataLoader(val_dataset_2, batch_size=batch_size, shuffle = True, pin_memory=False)
test_dataloader2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=batch_size, shuffle = False, pin_memory=False)
test_ts.append(test_t_2)
val_ts.append(len(val_dataloader2.dataset))

if deBug:
  print("train_x_2.shape = ", train_x_2.shape)
  print("train_y_2.shape = ", train_y_2.shape)
  print("test_x_2.shape  = ", test_x_2.shape)
  print("test_y_2.shape  = ", test_y_2.shape)
  print("test_t_2        = ", test_t_2)


#for round 3
j = 1
pct = 2036

# training dataset (formulate the training dataset by cutting off the original data and the cutoff[j*pct:] becaomes the test dataset)
train_x_3 = np.delete(data,   np.s_[j*pct:(j+1)*pct], axis=0)
train_y_3 = np.delete(target, np.s_[j*pct:(j+1)*pct], axis=0)

# test dataset (formulate the test dataet by taking the cutoff[j*pct:] when formulating the training dataset)
test_x_3  = data  [j*pct:(j+1)*pct]
test_y_3  = target[j*pct:(j+1)*pct]

test_t_3  = np.shape(test_y_3)[0] # time-dim of the test dataset


train_data_3 = OBSDataset(train_x_3, train_y_3)
N = len(train_data_3)
train_len = int(N * 0.8)

train_dataset_3 = torch.utils.data.Subset(train_data_3, list(range(0, train_len)))
val_dataset_3 = torch.utils.data.Subset(train_data_3, list(range(train_len, N)))

test_dataset_3 = OBSDataset(test_x_3, test_y_3)

train_dataloader3 = torch.utils.data.DataLoader(train_dataset_3, batch_size=batch_size, shuffle = True, pin_memory=False)
val_dataloader3 = torch.utils.data.DataLoader(val_dataset_3, batch_size=batch_size, shuffle = True, pin_memory=False)
test_dataloader3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=batch_size, shuffle = False, pin_memory=False)

test_ts.append(test_t_3)
val_ts.append(len(val_dataloader3.dataset))

if deBug:
  print("train_x_3.shape = ", train_x_3.shape)
  print("train_y_3.shape = ", train_y_3.shape)
  print("test_x_3.shape  = ", test_x_3.shape)
  print("test_y_3.shape  = ", test_y_3.shape)
  print("test_t_3        = ", test_t_3)

train_val_rounds = [[train_dataloader1, val_dataloader1], [train_dataloader2, val_dataloader2], [train_dataloader3, val_dataloader3]]
train_val_test_rounds = [[train_dataloader1, val_dataloader1, test_dataloader1], [train_dataloader2, val_dataloader2, test_dataloader2], [train_dataloader3, val_dataloader3, test_dataloader3]]

# ### Section C: CNN Architecture
# Basic CNN Framework

numChannels = 5

class CNN(torch.nn.Module):
	def __init__(self):

		super(CNN, self).__init__()
		self.conv11 = torch.nn.Conv2d(in_channels=numChannels, out_channels=30, kernel_size=(2,2), stride=(2,2))
		torch.nn.init.xavier_normal_(self.conv11.weight)
		self.relu11 = torch.nn.ReLU()

		self.conv12 = torch.nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(2,2), stride=(2,2))
		torch.nn.init.xavier_normal_(self.conv12.weight)
		self.relu12 = torch.nn.ReLU()

		self.conv13 = torch.nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(2,2), stride=(2,2))
		torch.nn.init.xavier_normal_(self.conv13.weight)
		self.relu13 = torch.nn.ReLU()
		
		self.flatten = torch.nn.Flatten()
        
		#note that flatten returns 30 * 3 * 22 = 1980 dimensions -- note: different from Shin!
		self.fc1 = torch.nn.Linear(in_features = 1980, out_features = 50)
		torch.nn.init.xavier_normal_(self.fc1.weight)
        
		self.relufc = torch.nn.ReLU()
        
		self.dropout1 = torch.nn.Dropout(0.5)

		self.fc2 = torch.nn.Linear(in_features = 50, out_features = 2)
		torch.nn.init.xavier_normal_(self.fc2.weight)

	def forward(self, x):
		#print(f"Initial shape x: {x.shape}")
		x = self.conv11(x)
		#print(f"Shape after conv11: {x.shape}")
		x = self.relu11(x)
		#print(f"Shape after relu11: {x.shape}")
		x = self.conv12(x)
		#print(f"Shape after conv12: {x.shape}")
		x = self.relu12(x)
		#print(f"Shape after relu12: {x.shape}")
		x = self.conv13(x)
		#print(f"Shape after conv13: {x.shape}")
		x = self.relu13(x)
		#print(f"Shape after relu13: {x.shape}")
		x = self.flatten(x)
		#print(f"Shape after flatten: {x.shape}")
		x = self.fc1(x)
		#print(f"Shape after fc1: {x.shape}")
		x = self.relufc(x)
		x = self.dropout1(x)
		x = self.fc2(x)
		#print(f"Shape after fc2: {x.shape}")
		return x

# We redefine same CNN architecture as the original model above.

#basic training

# Hyperparameters
lr = 1e-4
epochs = 100

# ### Seciton D/E: Compile and Training

#loop over rounds

compiled_train_losses = []
compiled_val_losses = []
for index, round in enumerate(train_val_rounds):
    train_losses = []
    val_losses = []
    train_dataloader = round[0]
    val_dataloader = round[1]
    #load model
    nn = torch.load("/home/bec32/project/ML_for_MJO/CNN/Ensemble_Results/Result_01/CNNMJO_Ben_MDL_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+".pth", weights_only=False).to(device)
    
    batches_per_datset = len(train_dataloader)
    print(f"batches per dataset: {batches_per_datset}")
    
    # send to GPU, if available
    # Optimizer, note that we use weight decay with coefficinet 1e-5
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr, weight_decay=1e-5)
    # Exponential Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001, factor=0.5, mode='min')
    loss_fn = torch.nn.MSELoss()

    # define early stopping class
    class EarlyStopping:
        def __init__(self, threshold, patience):
            self.threshold = threshold
            self.patience = patience
            self.patience_count = 0
        
        def check_early_stop(self, val_loss_history):
            #expect last item of val_loss_history to be most recent loss
            if len(val_loss_history) < self.patience:
                return False
        
            if min(val_loss_history[:-1]) - val_loss_history[-1] < self.threshold:
                self.patience_count += 1
                if self.patience_count >= self.patience:
                    return True
            else:
                self.patience_count = 0
                return False
        
    earlystopping = EarlyStopping(threshold = 0.0001, patience = 10) 
    
    best_validation_loss = float("inf")
    best_model = None
        
    #intial evaluation
    nn.eval()
    with torch.no_grad():
        total_train_loss = 0.0
        for batch_data, batch_target in train_dataloader:

            y = nn(batch_data)
            # Loss
            loss = loss_fn(y, batch_target)
            total_train_loss += loss.item() * batch_data.size(0)
        
        avg_train_loss = total_train_loss / len(train_dataloader.dataset)

     
        total_val_loss = 0.0
        for batch_data, batch_target in val_dataloader:

            pred_y = nn(batch_data)	
            loss = loss_fn(pred_y, batch_target)
            total_val_loss += loss.item() * batch_data.size(0)

        avg_val_loss = total_val_loss / len(val_dataloader.dataset)
        print(f"Performance Before Training: Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}")
    
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    
    
    for i in range(epochs):
        total_train_loss = 0.0
        for batch_data, batch_target in train_dataloader:
           
            nn.train()


            y = nn(batch_data)
            # Loss
            loss = loss_fn(y, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_data.size(0)
        avg_train_loss = total_train_loss / len(train_dataloader.dataset)

   
        nn.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_target in val_dataloader:

                pred_y = nn(batch_data)	
                loss = loss_fn(pred_y, batch_target)
                total_val_loss += loss.item() * batch_data.size(0)

        avg_val_loss = total_val_loss / len(val_dataloader.dataset)
        # progress scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch: {i + 1}, Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}, LR: {scheduler.get_last_lr()}")

        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            best_model = copy.deepcopy(nn)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
  
        if earlystopping.check_early_stop(val_losses):
            break

    nn = best_model
    nn.eval()
    torch.save(nn, "/home/bec32/project/ML_for_MJO/CNN/Ensemble_Results/Result_02/CNNMJO_Ben_MDL_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_round"+str(index + 1)+'.pth')
    compiled_train_losses.append(train_losses)
    compiled_val_losses.append(val_losses)


# ### Section F: Compute Final Predictions

dirSave = "/home/bec32/project/ML_for_MJO/CNN/Ensemble_Results/Result_03/"
for index, round in enumerate(train_val_test_rounds):
    print(f"Round {index+1}")

    train_dataloader = round[0]
    val_dataloader = round[1]
    test_dataloader = round[2]

    #load model
    nn = torch.load("/home/bec32/project/ML_for_MJO/CNN/Ensemble_Results/Result_02/CNNMJO_Ben_MDL_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_round"+str(index + 1)+'.pth', weights_only=False).to(device)
    nn.eval()
    
    all_train_preds = []
    all_train_targets = []
    all_val_preds = []
    all_val_targets = []
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for batch_data, batch_target in train_dataloader:

            y = nn(batch_data)
            
            all_train_preds.append(y.detach().cpu())
            all_train_targets.append(batch_target.detach().cpu())

        for batch_data, batch_target in val_dataloader:

            pred_y = nn(batch_data)	
            
            all_val_preds.append(pred_y.detach().cpu())
            all_val_targets.append(batch_target.detach().cpu())
        
        for batch_data, batch_target in test_dataloader:

            pred_y = nn(batch_data)	
            
            all_test_preds.append(pred_y.detach().cpu())
            all_test_targets.append(batch_target.detach().cpu())
         

    all_train_preds = torch.cat(all_train_preds, dim=0)
    all_train_targets = torch.cat(all_train_targets, dim=0)
    all_val_preds = torch.cat(all_val_preds, dim=0)
    all_val_targets = torch.cat(all_val_targets, dim=0)
    all_test_preds = torch.cat(all_test_preds, dim=0)
    all_test_targets = torch.cat(all_test_targets, dim=0)

    os.system("rm -rf " + dirSave + "CNNMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_rund"+str(index+1)+"_Prdct_Train.nc")
    # wrt to netCDF
    os.system("rm -rf " + dirSave + "CNNMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_rund"+str(index+1)+"_Prdct_Train.nc")
    create_NC = Dataset(dirSave+"CNNMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_rund"+str(index+1)+"_Prdct_Train.nc", "w", format="NETCDF4")
    create_NC.createDimension("time", tot - val_ts[index] - test_ts[index])
    create_NC.createDimension("var",  2)
    mjoTrainPred    = create_NC.createVariable("mjoTrainPred", "f4", ("time","var",))  # CSXM: mjoPred is the ctrb in Shin et al. (NPJCAS 2024)
    mjoTrainPred[:] = all_train_preds
    mjoTrainTagt    = create_NC.createVariable("mjoTrainTagt", "f4", ("time","var",))  # (ASXM)
    mjoTrainTagt[:] = all_train_targets


    create_NC.close()

    os.system("rm -rf " + dirSave + "CNNMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_rund"+str(index+1)+"_Prdct_Val.nc")
    # wrt to netCDF
    os.system("rm -rf " + dirSave + "CNNMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_rund"+str(index+1)+"_Prdct_Val.nc")
    create_NC = Dataset(dirSave+"CNNMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_rund"+str(index+1)+"_Prdct_Val.nc", "w", format="NETCDF4")
    create_NC.createDimension("time", val_ts[index])
    create_NC.createDimension("var",  2)
    mjoValPred    = create_NC.createVariable("mjoValPred", "f4", ("time","var",))  # CSXM: mjoPred is the ctrb in Shin et al. (NPJCAS 2024)
    mjoValPred[:] = all_val_preds
    mjoValTagt    = create_NC.createVariable("mjoValTagt", "f4", ("time","var",))  # (ASXM)
    mjoValTagt[:] = all_val_targets                                                         # (ASXM)

    create_NC.close()

    os.system("rm -rf " + dirSave + "CNNMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_rund"+str(index+1)+"_Prdct_Test.nc")
    # wrt to netCDF
    os.system("rm -rf " + dirSave + "CNNMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_rund"+str(index+1)+"_Prdct_Test.nc")
    create_NC = Dataset(dirSave+"CNNMJO_Ben_OBS_leadTm"+str(leadTm)+"_ensm"+str(seed_num)+"_rund"+str(index+1)+"_Prdct_Test.nc", "w", format="NETCDF4")
    create_NC.createDimension("time", test_ts[index])
    create_NC.createDimension("var",  2)
    mjoTestPred    = create_NC.createVariable("mjoTestPred", "f4", ("time","var",))  # CSXM: mjoPred is the ctrb in Shin et al. (NPJCAS 2024)
    mjoTestPred[:] = all_test_preds
    mjoTestTagt    = create_NC.createVariable("mjoTestTagt", "f4", ("time","var",))  # (ASXM)
    mjoTestTagt[:] = all_test_targets                                                         # (ASXM)

    create_NC.close()

print(f"\nCompleted: {__file__}")

        


                                                     


