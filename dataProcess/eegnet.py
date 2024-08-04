import torch
import torch.nn as nn
from utils import *

class EEGNet(nn.Module):
    def __init__(
            self, F1, D, F2=None, num_cls=5,
            in_channel=16, dropout=0.5
    ) -> None:
        super(EEGNet, self).__init__()

        if F2 is None:
            F2 = F1 * D

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d([31, 32, 0, 0]), nn.Conv2d(1, F1, (1, 64), bias=False), nn.BatchNorm2d(F1),
            MaxNormConstraintConv2d(F1, F1*D, (in_channel, 1), groups=F1, max_norm_value=1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(),
            nn.AvgPool2d((1,4)), nn.Dropout(dropout)
        )

        self.block_2 = nn.Sequential(
            nn.ZeroPad2d([7, 8, 0, 0]), nn.Conv2d(F1*D, F1*D, (1,16), groups=F1*D, bias=True),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False), 
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1,8)), nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), MaxNormConstraintLinear(F2*(1000//32), num_cls, max_norm_value=0.25)
        )

        self.input_bn = nn.BatchNorm2d(1)
    
    def forward(self, x, norm_len=1000):
        x = norm_length(x, norm_len)
        x = self.input_bn(x.unsqueeze(1))
        x = self.block_2(self.block_1(x))
        x = self.classifier(x)
        return x

class MaxNormConstraintConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm_value=1, norm_axis=2, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(
                torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True)
            )
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= desired / norms
        return w

class MaxNormConstraintLinear(nn.Linear):
    def __init__(self, *args, max_norm_value=1, norm_axis=0, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(
                torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True)
            )
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= desired / norms
        return w

if __name__ == '__main__':
    x = torch.randn([100, 6, 1024])
    net = EEGNet(F1=4, D=2, F2=8, in_channel=6, dropout=0.5)
    y = net(x)
    print(y.shape)