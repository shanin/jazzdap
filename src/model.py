import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import normalize

batch_size = 16
number_of_patches = 20
patch_size = 25
segment_length = 500  # number_of_patches x patch_size
feature_size = 301
number_of_classes = 62
step_notes = 5
SR = 22050
hop_size = 256
RNN = 'GRU'
number_of_channels = 1

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):

        print('x ', x.shape)
        # (samples * timesteps, input_size, channels)
        x_reshape = x.transpose(1, -1).reshape(-1, x.size(3), x.size(4), x.size(1)).transpose(2, 3).transpose(1, 2)
        
        print('x_reshape ', x_reshape.shape)
        
        y = self.module(x_reshape)
        print('x_hat ', y.shape)
        
        # (samples, timesteps, output_size)
        y = y.view(x.size(0), -1, *y.size()[1:]).transpose(1,2)
        print('y ', y.shape)

        return y

class CRNN(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # CNN PART
        #add correct padding mode
        #add kernel regularizer
        self.td1 = TimeDistributed(
            nn.Conv2d(
                1, 64, kernel_size = (1, 5), stride = (1, 5), 
                padding = (0,2), dtype=torch.float
            )
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.td2 = TimeDistributed(
            nn.Conv2d(
                64, 64, kernel_size = (3, 5), 
                padding = (1,2), dtype=torch.float
            )
        )
        self.bn2 = nn.BatchNorm3d(64)
        self.td3 = TimeDistributed(
            nn.Conv2d(
                64, 64, kernel_size = (3, 3), 
                padding = (1,1), dtype=torch.float
            )
        )
        self.bn3 = nn.BatchNorm3d(64)
        self.td4 = TimeDistributed(
            nn.Conv2d(
                64, 16, kernel_size = (3, 15), 
                padding = (1,7), dtype=torch.float
            )
        )
        self.bn4 = nn.BatchNorm3d(16)
        self.td5 = TimeDistributed(
            nn.Conv2d(
                16, 1, kernel_size = (1, 1), 
                dtype=torch.float
            )
        )
        
        # RNN PART
        #add regularizer
        self.rnn = nn.GRU(61, 128, bidirectional = True)
        self.classifier = nn.Linear(256, 62)


    def forward(self, x):
        
        samples = x.size(0)
        x = F.relu(self.td1(x))
        x = self.bn1(x)
        x = F.relu(self.td2(x))
        x = self.bn2(x)
        x = F.relu(self.td3(x))
        x = self.bn3(x)
        x = F.relu(self.td4(x))
        x = self.bn4(x)
        x = F.relu(self.td5(x))
        x = x.reshape(x.size(0), -1, x.size(4))
        x = self.rnn(x)[0]
        x = self.classifier(x.view(-1, x.size(-1)))
        x = x.reshape(samples, -1, x.size(-1))
        
        return x

