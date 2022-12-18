"""
King's College scene from Cambridge Landmarks, a large scale outdoor visual relocalisation dataset taken around Cambridge University. Contains original video, with extracted image frames labelled with their 6-DOF camera pose and a visual reconstruction of the scene. 

- https://www.repository.cam.ac.uk/handle/1810/251342
"""

from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch
import os
import glob
import collections
from itertools import combinations
from scipy.spatial.transform import Rotation as Rot
from PIL import Image


class CambridgeLandmarkDataset(Dataset):

    def __init__(self, data_root_dir, split, random_crop = True):
        self.data_root_dir = data_root_dir
        self.transforms = T.Compose([T.RandomCrop((224, 224)),
        					  T.ToTensor(),
						  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
								    
        self.random_crop = random_crop
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        lines = []
        with open(os.path.join(data_root_dir, 'dataset_{}.txt'.format(split)), 'r') as f:
            lines = f.readlines()
            
        self.dict = collections.defaultdict(list)
        for line in lines[3:]:
            seq_num = line.split('/')[0]
            self.dict[seq_num].append(line)
        
        self.input_pairs = []
        
        for key in self.dict:
            new_pairs = list(combinations(self.dict[key], 2))
            valid_pair_ids = []
            for i in range(len(new_pairs)):
                data_1 = int(new_pairs[i][0].split(' ')[0][10:15])
                data_2 = int(new_pairs[i][1].split(' ')[0][10:15])
                if(abs(data_1 - data_2) <= 10):
                    valid_pair_ids.append(i)
                    
            for idx in valid_pair_ids:
                self.input_pairs.append(new_pairs[idx])
                
        print(self.__len__())
            
        
    def __len__(self):
        return len(self.input_pairs)
        
    def __getitem__(self, idx):
        image_1_data = self.input_pairs[idx][0].split(' ')
        image_2_data = self.input_pairs[idx][1].split(' ')
        
        image_1_name = os.path.join(self.data_root_dir, image_1_data[0])
        image_2_name = os.path.join(self.data_root_dir, image_2_data[0])
        
        image_1 = Image.open(image_1_name)
        image_2 = Image.open(image_2_name)
        
        image_1_input = self.transforms(image_1).to(self.device)
        image_2_input = self.transforms(image_2).to(self.device)
        
        R_w_c1 = Rot.from_quat([float(image_1_data[4]), float(image_1_data[5]), float(image_1_data[6]), float(image_1_data[7])]).as_matrix()	
        R_w_c2 = Rot.from_quat([float(image_2_data[4]), float(image_2_data[5]), float(image_2_data[6]), float(image_2_data[7])]).as_matrix()
        t_w_c1 = [float(image_1_data[1]), float(image_1_data[2]), float(image_1_data[3])]
        t_w_c2 = [float(image_2_data[1]), float(image_2_data[2]), float(image_2_data[3])]
        
        T_w_c1 = np.zeros([4,4])
        T_w_c1[0:3,0:3] = R_w_c1
        T_w_c1[0:3,3] = t_w_c1
        T_w_c1[3,3] = 1.0
        
        T_w_c2 = np.zeros([4,4])
        T_w_c2[0:3,0:3] = R_w_c2
        T_w_c2[0:3,3] = t_w_c2
        T_w_c2[3,3] = 1.0
        
        T_c1_c2 = np.matmul(np.linalg.inv(T_w_c1), T_w_c2)
        t_c1_c2 = list(T_c1_c2[0:3,3])
        R_c1_c2 = T_c1_c2[0:3,0:3]
        q_c1_c2 = Rot.from_matrix([R_c1_c2]).as_quat()[0]
        pose = torch.hstack((torch.tensor(t_c1_c2).to(self.device), torch.tensor(q_c1_c2).to(self.device)))
        
        sample_pair = {'input': [image_1_input, image_2_input], 'label': pose}
        
        return sample_pair
        
#dataset = CambridgeLandmarkDataset(data_root_dir, "train", 8)
#dataset.__getitem__(1)

            
        
        
