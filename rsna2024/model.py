import torch
import torch.nn as nn

class Model(nn.Module):
    def ConvLayer(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
    
    def FcLayer(self, in_dim, out_dim, bias=True):
        return nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias),
            nn.ReLU(),
        )

    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = self.ConvLayer(1, 32)
        self.layer2 = self.ConvLayer(32, 64)
        self.layer3 = self.ConvLayer(64, 128)
        self.layer4 = self.ConvLayer(128, 256)
        self.layer5 = self.ConvLayer(256, 256)
        
        self.fc1 = self.FcLayer(7* 7* 256, 2048)
        self.fc2 = self.FcLayer(2048, 64)
        self.fc3 = nn.Linear(64, 3, bias=False)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(-1, 7* 7* 256)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x