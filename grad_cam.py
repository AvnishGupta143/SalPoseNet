from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models.baseline_model import BaselineGoogleNetModel, SaliencyBaselineGoogleNetExpModel
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader

import torch
from datasets.kitti_odom_saliency_dataset import KittiOdomSaliencyDataset


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
MODEL_PATH = "/home/pear_group/avnish_ws/PEAR_LAB/model_weights/IM_IM_Sal_Opt_kitti_01_learn_beta/model_60.pth" 

model = SaliencyBaselineGoogleNetExpModel()    
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH), strict=False)
model.eval()

data_root_dir = "/home/pear_group/avnish_ws/PEAR_LAB/data/Kitti_odo"
eval_dataset = KittiOdomSaliencyDataset(data_root_dir, [0], "train")
data_loader = DataLoader(eval_dataset, batch_size = 1, shuffle = False, drop_last = False)  

target_layers = [model.features_extractor]

data = next(iter(data_loader))
input_tensor = data['input'][0]
print(input_tensor.shape)
# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

class FasterRCNNBoxScoreTarget:
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, model_outputs):
        return output

targets = FasterRCNNBoxScoreTarget(labels = data['label'])

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
#grayscale_cam = grayscale_cam[0, :]
#visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
