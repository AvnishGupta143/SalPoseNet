
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch
import os
from glob import glob
import collections
from itertools import combinations
from scipy.spatial.transform import Rotation as Rot
from PIL import Image

class KittiOdomSaliencyEvalDataset(Dataset):

    def __init__(self, data_root_dir, seqs, split):
        self.data_input_dir = os.path.join(data_root_dir, 'data_odometry_color', 'dataset')
        self.data_label_dir = os.path.join(data_root_dir, 'data_odometry_poses', 'dataset')
        self.img_height = 224
        self.img_width = 224
        self.seqs = seqs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.collect_frames()
        
        self.transform = T.Compose([
            			T.Resize((224, 224)),
        			T.ToTensor(), 
        			T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]) #imagenet normalization stats
        			
        self.trasnform_sal = T.Compose([T.Resize((224, 224)),
        			       T.ToTensor()])
       
    def collect_frames(self):
        self.input_pairs = []
        self.labels = []
        
        for seq in self.seqs:
            with open(os.path.join(self.data_label_dir, 'poses', '%.2d' % seq + '.txt'), 'r') as f:
                lines = f.readlines()
                self.labels.append(lines)
        
                print(len(lines))
                print(len(self.labels))
        self.dict = collections.defaultdict(list)
        for seq in self.seqs:
            seq_dir = os.path.join(self.data_input_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_2')
            images = sorted(glob(img_dir + '/*.png'))
            N = len(images)
            for n in range(N):
                seq_str = '%.2d' % seq
                self.dict[seq].append('%.2d %.6d' % (seq, n) + ' ' + images[n]) #seq_num, image_num, image_path  
        
        for key in self.dict:
            new_pairs = list(combinations(self.dict[key], 2))
            valid_pair_ids = []
            for i in range(len(new_pairs)):
                data_1 = int(new_pairs[i][0].split(' ')[1])
                data_2 = int(new_pairs[i][1].split(' ')[1])
                if(abs(data_1 - data_2) == 1 and data_1 != data_2):
                    valid_pair_ids.append(i)
                    
            for idx in valid_pair_ids:
                self.input_pairs.append(new_pairs[idx])
                data_1 = int(new_pairs[i][0].split(' ')[0])
                data_2 = int(new_pairs[i][1].split(' ')[0])
                if(data_1 == data_2): 
                    pass
               
        print(self.__len__())
            
        
    def __len__(self):
        return len(self.input_pairs)
        
    def __getitem__(self, idx):
        image_1_data = self.input_pairs[idx][0].split(' ')
        image_2_data = self.input_pairs[idx][1].split(' ')
        
        image_1_name = image_1_data[2]
        image_2_name = image_2_data[2]
        
        image_1_c = Image.open(image_1_name)
        image_2_c = Image.open(image_2_name)
        
        image_1_sal = Image.open(image_1_name.replace("image_2", "image_2_sal"))
        image_2_sal = Image.open(image_2_name.replace("image_2", "image_2_sal"))
        
        image_1_input_c = self.transform(image_1_c).to(self.device)
        image_2_input_c = self.transform(image_2_c).to(self.device)
        
        image_1_input_sal = self.trasnform_sal(image_1_sal).to(self.device)
        image_2_input_sal = self.trasnform_sal(image_2_sal).to(self.device)
        
        image_1_input = torch.vstack((image_1_input_c, image_1_input_sal))
        image_2_input = torch.vstack((image_2_input_c, image_2_input_sal))
        
        seq1 = int(image_1_data[0])
        img1_n = int(image_1_data[1])
        T_w_c1 = np.vstack([np.array(self.labels[0][img1_n].split(' ')).reshape((3,4)).astype(np.float32), [0, 0, 0, 1]])
        #t_w_c1 = list(T_w_c1[0:3,3])
        #R_w_c1 = T_w_c1[0:3,0:3]
        #q_w_c1 = Rot.from_matrix([R_w_c1]).as_quat()[0]
        #pose_w_c1 = np.hstack((t_w_c1, q_w_c1))

        seq2 = int(image_2_data[0])
        img2_n = int(image_2_data[1])
        T_w_c2 = np.vstack([np.array(self.labels[0][img2_n].split(' ')).reshape((3,4)).astype(np.float32), [0, 0, 0, 1]])
        t_w_c2 = list(T_w_c2[0:3,3])
        R_w_c2 = T_w_c2[0:3,0:3]
        q_w_c2 = Rot.from_matrix([R_w_c2]).as_quat()[0]
        pose_w_c2 = np.hstack((t_w_c2, q_w_c2))
        
        sample_pair = {'input': [image_1_input, image_2_input], 'labels': pose_w_c2, 'frame_1_T': T_w_c1, 'frame_2_T': T_w_c2, 'frame1_num': img1_n, 'frame2_num': img2_n}
        
        if(False):
            print("----------IMAGE1----------")
            print(image_1_data)
            print(T_w_c1)
            
            print("----------IMAGE2----------")
            print(image_2_data)
            print(T_w_c2)
            
        return sample_pair

class KittiOdomSaliencyDataset(Dataset):

    def __init__(self, data_root_dir, seqs, split):
        self.data_input_dir = os.path.join(data_root_dir, 'data_odometry_color', 'dataset')
        self.data_label_dir = os.path.join(data_root_dir, 'data_odometry_poses', 'dataset')
        self.img_height = 224
        self.img_width = 224
        self.seqs = seqs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.collect_frames()
        
        self.transform = T.Compose([
            			T.Resize((224, 224)),
        			T.ToTensor(), 
        			T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]) #imagenet normalization stats
        			
        self.trasnform_sal = T.Compose([T.Resize((224, 224)),
        			       T.ToTensor()])
        			
        			
      
        """
        if(split == "train"):
            self.transform = custom_transforms.Compose(
                              [
                                custom_transforms.RandomHorizontalFlip(),
                                custom_transforms.ArrayToTensor(),
                                normalize
                              ])
        else:
            self.transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
        """
       
    def collect_frames(self):
        self.input_pairs = []
        self.labels = []
        
        for seq in self.seqs:
            with open(os.path.join(self.data_label_dir, 'poses', '%.2d' % seq + '.txt'), 'r') as f:
                lines = f.readlines()
                self.labels.append(lines)
        
        self.dict = collections.defaultdict(list)
        for seq in self.seqs:
            seq_dir = os.path.join(self.data_input_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_2')
            images = sorted(glob(img_dir + '/*.png'))
            N = len(images)
            for n in range(N):
                seq_str = '%.2d' % seq
                self.dict[seq].append('%.2d %.6d' % (seq, n) + ' ' + images[n]) #seq_num, image_num, image_path  
        
        for key in self.dict:
            new_pairs = list(combinations(self.dict[key], 2))
            valid_pair_ids = []
            for i in range(len(new_pairs)):
                data_1 = int(new_pairs[i][0].split(' ')[1])
                data_2 = int(new_pairs[i][1].split(' ')[1])
                if(abs(data_1 - data_2) <= 5 and data_1 != data_2):
                    valid_pair_ids.append(i)
                    
            for idx in valid_pair_ids:
                self.input_pairs.append(new_pairs[idx])
                data_1 = int(new_pairs[i][0].split(' ')[0])
                data_2 = int(new_pairs[i][1].split(' ')[0])
                if(data_1 == data_2): 
                    pass
                
                
        print(self.__len__())
            
        
    def __len__(self):
        return len(self.input_pairs)
        
    def __getitem__(self, idx):
        image_1_data = self.input_pairs[idx][0].split(' ')
        image_2_data = self.input_pairs[idx][1].split(' ')
        
        image_1_name = image_1_data[2]
        image_2_name = image_2_data[2]
        
        image_1_c = Image.open(image_1_name)
        image_2_c = Image.open(image_2_name)
        
        image_1_sal = Image.open(image_1_name.replace("image_2", "image_2_sal"))
        image_2_sal = Image.open(image_2_name.replace("image_2", "image_2_sal"))
        
        image_1_input_c = self.transform(image_1_c).to(self.device)
        image_2_input_c = self.transform(image_2_c).to(self.device)
        
        image_1_input_sal = self.trasnform_sal(image_1_sal).to(self.device)
        image_2_input_sal = self.trasnform_sal(image_2_sal).to(self.device)
        
        image_1_input = torch.vstack((image_1_input_c, image_1_input_sal))
        image_2_input = torch.vstack((image_2_input_c, image_2_input_sal))
        
        
        seq1 = int(image_1_data[0])
        img1_n = int(image_1_data[1])
        T_w_c1 = np.vstack([np.array(self.labels[seq1][img1_n].split(' ')).reshape((3,4)).astype(np.float32), [0, 0, 0, 1]])

        seq2 = int(image_2_data[0])
        img2_n = int(image_2_data[1])
        T_w_c2 = np.vstack([np.array(self.labels[seq2][img2_n].split(' ')).reshape((3,4)).astype(np.float32), [0, 0, 0, 1]])

        T_c1_c2 = np.matmul(np.linalg.inv(T_w_c1), T_w_c2)
        t_c1_c2 = list(T_c1_c2[0:3,3])
        R_c1_c2 = T_c1_c2[0:3,0:3]
        q_c1_c2 = Rot.from_matrix([R_c1_c2]).as_quat()[0]
        pose = torch.hstack((torch.tensor(t_c1_c2).to(self.device), torch.tensor(q_c1_c2).to(self.device)))
        
        sample_pair = {'input': [image_1_input, image_2_input], 'label': pose}
        
        if(False):
            print("----------IMAGE1----------")
            print(image_1_data)
            print(T_w_c1)
            
            print("----------IMAGE2----------")
            print(image_2_data)
            print(T_w_c2)
            
        return sample_pair
        
#dataset = KittiOdomDataset("/home/pear_group/avnish_ws/PEAR_LAB/data/Kitti_odo", [0, 1], "train")
#dataset.__getitem__(1)

            
        
        
