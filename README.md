# SalPoseNet

We present SalPoseNet, a joint end-to-end deep learning framework using visual saliency for relative 6 degrees of freedom (DoF) camera pose estimation between sequences of frames. Our experiments show that using visual saliency along with the camera images can help in making much more reasonable pose predictions in comparison to other methods using only camera images. The SalPoseNet network shares a common backbone for extracting features from the previous and current image frames which makes it faster and smaller in size. We were able to obtain 1m and 1\textdegree accuracy on large scale outdoor scenes. We trained and tested our algorithm on the KITTI odometry dataset. Our algorithm is self-supervised in a way that we generate labels automatically from the large scale publicly available odometry datasets.

###

### Training 

1. Change the dataset and the model type in the train.py script.
2. To generate data for saliency, change the sequence you want to generate for saliency data for in the script convert_saliency_data.py and run
'''
python convert_saliency_data.py
'''
4. Run the following command to start training
'''
python train.py
'''

### Testing

1. Change the dataset and the model type in the evaluate.py script.
2. Run the following command to generate evaluation
'''
python evaluate.py
'''

### Generating the plots
1. After evaluation we will get two files, stamped_groundtruth.txt and stamped_traj_estimate.txt. 
2. Run rpg_trajectory_evalution package for generating the RMSE ATE and trajectory plots on the above files. 

