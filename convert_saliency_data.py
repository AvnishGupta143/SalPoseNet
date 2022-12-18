import os
import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from saliency_model.utils.data_process import preprocess_img, postprocess_img
from PIL import Image
from glob import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

flag = 0 # 0 for TranSalNet_Dense, 1 for TranSalNet_Res

if flag:
    from saliency_model.TranSalNet_Res import TranSalNet
    model = TranSalNet()
    model.load_state_dict(torch.load("/home/pear_group/avnish_ws/PEAR_LAB/model_weightsTranSalNet_Res.pth"))
else:
    from saliency_model.TranSalNet_Dense import TranSalNet
    model = TranSalNet()
    model.load_state_dict(torch.load("/home/pear_group/avnish_ws/PEAR_LAB/model_weights/TranSalNet_Dense.pth"))

model = model.to(device) 
model.eval()

seq = 2
data_root_dir = "/home/pear_group/avnish_ws/PEAR_LAB/data/Kitti_odo"
data_input_dir = os.path.join(data_root_dir, 'data_odometry_color', 'dataset')
seq_dir = os.path.join(data_input_dir, 'sequences', '%.2d' % seq)
in_img_dir = os.path.join(seq_dir, 'image_2')
out_img_dir = os.path.join(seq_dir, 'image_2_sal')
images = sorted(glob(in_img_dir + '/*.png'))

for im in images:
    out_image = os.path.join(out_img_dir, im.split("/")[-1])
    img = preprocess_img(im) # padding and resizing input image into 384x288
    img = np.array(img)/255.
    img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)
    img = torch.from_numpy(img)
    img = img.type(torch.cuda.FloatTensor).to(device)
    pred_saliency = model(img)
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency.squeeze())

    pred_saliency = postprocess_img(pic, im) # restore the image to its original size as the result

    cv2.imwrite(out_image, pred_saliency, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

