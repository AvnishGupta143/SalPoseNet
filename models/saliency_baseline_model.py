import torch
import torchvision
from torchvision.models import googlenet, GoogLeNet_Weights
from torchsummary import summary
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class SaliencyBaselineGoogleNetModel(torch.nn.Module):
    def __init__(self, num_features = 512):
        super(BaselineGoogleNetModel, self).__init__()
        weights = GoogLeNet_Weights.DEFAULT
        google_lenenet_model = googlenet(weights=weights)
        self.features_extractor = torch.nn.Sequential(*list(google_lenenet_model.children())[1:-2])
        
        self.init_conv = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        
        self.fc1 = torch.nn.Linear(2048, 512)
        self.activation = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, num_features)
        
        # Translation
        self.fc_xyz = torch.nn.Linear(num_features, 3, bias = True)

        # Rotation in quaternions
        self.fc_quat = torch.nn.Linear(num_features, 4, bias = True)
        
        
        init_modules = [self.fc_xyz, self.fc_quat]
        
        for m in init_modules:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)
                    
    def forward(self, x):
        feature_image_1 = torch.squeeze(self.features_extractor(self.init_conv(x[0])))
        feature_image_2 = torch.squeeze(self.features_extractor(self.init_conv(x[1])))
        
        combined_features = torch.cat((feature_image_1, feature_image_2), dim = -1)
        out = self.fc1(combined_features)
        out = self.activation(out)
        out = self.fc2(out)
        
        x_translations = torch.squeeze(self.fc_xyz(out)) 
        x_rotations = torch.squeeze(self.fc_quat(out))
        
        x_poses = torch.cat((x_translations, x_rotations), dim = -1)
        return x_poses
