import torch
import torch.nn.functional as F

class PoseNetCriterion(torch.nn.Module):
    def __init__(self, beta = 512.0, learn_beta = True, sx = 0.0, sq = 0.0):
        super(PoseNetCriterion, self).__init__()
        self.loss_fn = torch.nn.L1Loss()
        self.learn_beta = learn_beta
        
        if not self.learn_beta:
            self.sx = 0.0
            self.sq = 0.0
            
        self.sx = torch.nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = torch.nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)

    def forward(self, x, y):
        """
        Args:
            x: list(N x 7, N x 7) - prediction (xyz, quat)
            y: list(N x 7, N x 7) - target (xyz, quat)
        """

        loss = 0
        
        pred_x = x[:, :3]
        target_x = y[:, :3]
        
        pred_q = x[:, 3:]
        pred_q = F.normalize(pred_q, p = 2, dim = -1)
        target_q = y[:, 3:]
        
        # Translation loss
        loss_x = torch.exp(-self.sx) * self.loss_fn(pred_x, target_x) + self.sx
        
        # Rotation loss
        loss_q = torch.exp(-self.sq) * self.loss_fn(pred_q, target_q) + self.sq
        
        loss = loss_q + loss_x

        return loss
