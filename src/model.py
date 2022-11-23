import torch
import torch.nn as nn
import torch.nn.functional as F

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

        n = x.size(0)
        # NCTWH -> NTCWH -> (NT)CWH
        x = x.transpose(1,2)
        x = x.reshape(-1, *x.shape[2:])
        y = self.module(x)
        #(NT)CWH -> NTCWH -> NCTWH
        y = y.reshape(n, -1, *y.shape[1:]).transpose(1,2)

        return y

class CRNN(nn.Module):

    def __init__(self):
        super(CRNN, self).__init__()
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
        self.rnn = nn.GRU(61, 128, bidirectional = True, batch_first=True)
        self.classifier = nn.Linear(256, 63)


    def forward(self, x, debug = False):

        x = F.relu(self.td1(x))
        x = self.bn1(x)
        x = F.relu(self.td2(x))
        x = self.bn2(x)
        x = F.relu(self.td3(x))
        x = self.bn3(x)
        x = F.relu(self.td4(x))
        x = self.bn4(x)
        x = F.relu(self.td5(x))

        #N1TWH -> N(TW)H
        x = x.reshape(x.size(0), -1, x.size(4))
        x = self.rnn(x)[0]
        x = self.classifier(x)
        x = x.reshape(x.size(0), 20, 25, 63)
        return x



class CNN_base(nn.Module):

    def __init__(self):
        super(CNN_base, self).__init__()
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
        

        self.classifier = nn.Linear(61, 63)


    def forward(self, x, debug = False):

        x = F.relu(self.td1(x))
        x = self.bn1(x)
        x = F.relu(self.td2(x))
        x = self.bn2(x)
        x = F.relu(self.td3(x))
        x = self.bn3(x)
        x = F.relu(self.td4(x))
        x = self.bn4(x)
        x = F.relu(self.td5(x))

        #N1TWH -> N(TW)H
        x = x.reshape(x.size(0), -1, x.size(4))

        x = self.classifier(x)
        x = x.reshape(x.size(0), 20, 25, 63)
        return x