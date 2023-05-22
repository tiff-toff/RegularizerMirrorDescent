import torch
import torch.nn as nn

class RMD_Loss(nn.Module):
    def __init__(self, sample_loss_func, num_datapoints, reduction='mean'):
        super(RMD_Loss, self).__init__()
        self.sample_loss_func = sample_loss_func
        self.z = torch.zeros(num_datapoints)
        self.idx = 0
        self.reduction = reduction
    
    def set_z_values(self, z, idx):
        self.z = z
        self.idx = idx

    def forward(self, predictions, target):
        sample_loss = self.sample_loss_func(predictions, target)
        # if (sample_loss.isnan().any()):
        #     torch.set_printoptions(threshold=200)
        #     print("pred", predictions)
        #     print("target", target)
        #     print("loss", sample_loss)

        adjusted_sample_loss = torch.sqrt(2 * sample_loss)
        squared_loss = 0.5 * torch.square(self.z[self.idx] - adjusted_sample_loss)
        
        if self.reduction == 'mean':
            return torch.mean(squared_loss)
        elif self.reduction == 'sum':
            return torch.sum(squared_loss)
        else:
            return squared_loss

