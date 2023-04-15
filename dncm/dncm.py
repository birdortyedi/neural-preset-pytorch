import torch
from torch import nn
from torchvision import models
from kornia.geometry.transform import resize
from kornia.enhance.normalize import Normalize


class DNCM(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.empty((3, k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((k, 3)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.Q)
        self.k = k
        
    def forward(self, I, T):
        bs, _, H, W = I.shape
        x = torch.flatten(I, start_dim=2).transpose(1, 2)
        out = x @ self.P @ T.view(bs, self.k, self.k) @ self.Q
        out = out.view(bs, H, W, -1).permute(0, 3, 1, 2)
        return out
    

class Encoder(nn.Module):
    def __init__(self, sz, k) -> None:
        super().__init__()
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.D = nn.Linear(in_features=1000, out_features=k*k)
        self.S = nn.Linear(in_features=1000, out_features=k*k)
        self.sz = sz
        
    def forward(self, I):
        I_theta = resize(I, self.sz, interpolation='bilinear')
        out = self.backbone(self.normalizer(I_theta))
        d = self.D(out)
        s = self.S(out)
        return d, s
        
    
if __name__ == "__main__":
    k = 32 
    I = torch.rand((8, 3, 1024, 1024)).cuda()
    net = DNCM(k).cuda()
    E = Encoder(256, k).cuda()
    d, s = E(I)
    out = net(I, s)
    print(out.shape)