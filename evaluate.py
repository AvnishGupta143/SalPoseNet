import torch
import numpy as np
from models.baseline_model import BaselineGoogleNetModel, SaliencyBaselineGoogleNetModel
from datasets.kitti_odom_dataset import KittiOdomEvalDataset
from datasets.kitti_odom_saliency_dataset import KittiOdomSaliencyEvalDataset
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import DataLoader
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

MODEL_PATH = "/home/pear_group/avnish_ws/PEAR_LAB/model_weights/IM_IM_Sal_Opt_kitti_01_learn_beta/model_60.pth" 

model = None
model_t = "baseline_sal" 
if model_t == "baseline":
    model = BaselineGoogleNetModel()
elif model_t == "baseline_sal":
    model = SaliencyBaselineGoogleNetModel()

model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

data_root_dir = "/home/pear_group/avnish_ws/PEAR_LAB/data/Kitti_odo"
eval_dataset = None
dataset = "Kitti_odom_sal"
                                          
elif(dataset == "Kitti_odom"):
    eval_dataset = KittiOdomDataset(data_root_dir, [0], "train")
    
elif(dataset =="Kitti_odom_sal"):
    eval_dataset = KittiOdomSaliencyDataset(data_root_dir, [0], "train")

data_loader = DataLoader(eval_dataset, batch_size = 1, shuffle = False, drop_last = False)  
 
with open('stamped_gt.txt', 'w') as gt_testfile:
    gt_testfile.write('# timestamp tx ty tz qx qy qz qw \n')
with open('stamped_pred.txt', 'w') as pred_testfile:
    pred_testfile.write('# timestamp tx ty tz qx qy qz qw \n')

for i, item in enumerate(data_loader):
    with torch.no_grad():
        

        if(i != item['frame1_num'] or i+1 != item['frame2_num']): 
           print("data_load_error")
           exit()
           
        predictions = model(item['input'])
        pred_t_c1_c2 = predictions[:3].detach().cpu().numpy()
        pred_t_c1_c2[2] = pred_t_c1_c2[2]/100.0
        pred_q_c1_c2 = F.normalize(predictions[3:], p = 2, dim = -1).detach().cpu().numpy()
        pred_rot_c1_c2 = Rot.from_quat([pred_q_c1_c2]).as_matrix()
        pred_T_c1_c2 = np.zeros([4,4])
        pred_T_c1_c2[0:3, 0:3] = pred_rot_c1_c2
        pred_T_c1_c2[0:3,3] = pred_t_c1_c2
        pred_T_c1_c2[3,3] = 1        
        
        true_T_w_c1 = np.squeeze(item['frame_1_T'].detach().cpu().numpy())
        
        pred_T_w_c2 = np.matmul(true_T_w_c1, pred_T_c1_c2)
        pred_t_w_c2 = list(pred_T_w_c2[0:3,3])
        pred_R_w_c2 = pred_T_w_c2[0:3,0:3]
        pred_q_w_c2 = Rot.from_matrix([pred_R_w_c2]).as_quat()[0]
        
        pred_pose_w_c2 = np.hstack((pred_t_w_c2, pred_q_w_c2))
        true_pose_w_c2 = item['labels'].detach().cpu().numpy().squeeze()
      	
        with open('stamped_groundtruth.txt', 'a') as gt_testfile:
            gt_testfile.write(str(i) + ' ' +  ' '.join([str(a) for a in true_pose_w_c2]) + '\n')
            
        with open('stamped_traj_estimate.txt', 'a') as pred_testfile:
            pred_testfile.write(str(i) + ' ' +  ' '.join([str(a) for a in pred_pose_w_c2]) + '\n')
        
        
        
        
    
