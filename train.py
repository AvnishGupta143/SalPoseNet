import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchsummary import summary

from datasets.cambridge_landmarks_dataset import CambridgeLandmarkDataset
from datasets.kitti_odom_dataset import KittiOdomDataset
from datasets.kitti_odom_saliency_dataset import KittiOdomSaliencyDataset

from models.baseline_model import BaselineGoogleNetModel, SaliencyBaselineGoogleNetModel
from losses.poseNetLoss import PoseNetCriterion

import wandb
wandb.init(project="Relative pose slam", entity="avnish1433")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

batch_size = 192

NUM_EPOCHS = 151
LR = 0.00001


model = None
model_t = "baseline_sal" 
if model_t == "baseline":
    model = BaselineGoogleNetModel()
elif model_t == "baseline_sal":
    model = SaliencyBaselineGoogleNetModel()

model = model.to(device)
print(model)

wandb.config = {
  "learning_rate": LR,
  "epochs": NUM_EPOCHS,
  "batch_size": batch_size
}

train_dataset = None
train_loader = None
val_dataset = None
val_loader = None
dataset = "Kitti_odom_sal"

if(dataset == "Cambridge"):
    data_root_dir = "/home/pear_group/avnish_ws/PEAR_LAB/data/KingsCollege"   
    train_dataset = CambridgeLandmarkDataset(data_root_dir, "train")
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last=True)        
    val_dataset = CambridgeLandmarkDataset(data_root_dir, "test")
    val_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last=True)    
                                          
elif(dataset == "Kitti_odom"):
    data_root_dir = "/home/pear_group/avnish_ws/PEAR_LAB/data/Kitti_odo"
    train_dataset = KittiOdomDataset(data_root_dir, [0], "train")
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last=True)        
    val_dataset = KittiOdomDataset(data_root_dir, [2],  "test")
    val_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last=True)   
    
elif(dataset =="Kitti_odom_sal"):
    data_root_dir = "/home/pear_group/avnish_ws/PEAR_LAB/data/Kitti_odo"
    train_dataset = KittiOdomSaliencyDataset(data_root_dir, [0], "train")
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last=True)        
    val_dataset = KittiOdomSaliencyDataset(data_root_dir, [2],  "test")
    val_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last=True)   
    
                                   
# Criterion
criterion = PoseNetCriterion(beta = 512, learn_beta = True)
criterion.to(device)

# Add all params for optimization
param_list = [{'params': model.parameters()}]
if criterion.learn_beta:
    param_list.append({'params': criterion.parameters()})


# Create optimizer
optimizer = optim.Adam(params = param_list, lr = LR, weight_decay=0.0005)

# Create Scheduler 
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,140], gamma=0.1)

for epoch in range(NUM_EPOCHS):
    # training phase
    epoch_loss = 0
    count_train_batches = 0
    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        predictions = model(data['input'])      
        loss = criterion(predictions, data['label'])
        epoch_loss += loss.item()
        batch_loss = loss.item()
        loss.backward()
        optimizer.step()
        count_train_batches+=1
        print("Epoch {} Batch {} Loss {}".format(epoch, count_train_batches, batch_loss))

    epoch_loss = epoch_loss / count_train_batches
    wandb.log({"train_loss": epoch_loss, "epoch":epoch})
    
    scheduler.step()

    # Validation phase
    if(epoch % 5 == 0):
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            count_val_batches = 0
            for idx, batch in enumerate(val_loader):
                inputs, labels = batch
                count_val_batches += 1
                predictions = model(data['input'])
                val_loss += criterion(predictions, data['label']).item()
            
            val_loss = val_loss / (count_val_batches)
            wandb.log({"val_loss": val_loss, "epoch":epoch})
    
    if(epoch%10 == 0):
        save_path = "/home/pear_group/avnish_ws/PEAR_LAB/model_weights/IM_IM_Sal_Opt_kitti_01_learn_beta/" + 'model_{}'.format(epoch) + '.pth'
        torch.save(model.state_dict(), save_path)

    # Optional
    wandb.watch(model)
    
    print("----------- Epoch {} Loss {} LR {}----------".format(epoch+1 , batch_loss, optimizer.param_groups[0]['lr']))



