import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):

    def _parse_config(self, config):
        self.n_classes = config['crnn'].get('number_of_classes', 62)
        self.patch_size = config['crnn'].get('patch_size', 25)
        self.number_of_patches = config['crnn'].get('number_of_patches', 20)

    def __init__(self, config):
        super(CRNN, self).__init__()
        self._parse_config(config)
        self.conv1 = nn.Conv2d(
                1, 64, kernel_size = (1, 5), stride = (1, 5), 
                padding = (0,2), dtype=torch.float
            )
        
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
                64, 64, kernel_size = (3, 5), 
                padding = (1,2), dtype=torch.float
            )
        
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(
                64, 64, kernel_size = (3, 3), 
                padding = (1,1), dtype=torch.float
            )
        
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                64, 16, kernel_size = (3, 15), 
                padding = (1,7), dtype=torch.float
            )
        
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(
                16, 1, kernel_size = (1, 1), 
                dtype=torch.float
            )
        
        
        # RNN PART
        self.rnn = nn.GRU(61, 128, bidirectional = True, batch_first=True)
        self.classifier = nn.Linear(256, self.n_classes)


    def forward(self, x):

        n = x.size(0)
        # NCTWH -> NTCWH -> (NT)CWH
        x = x.transpose(1,2)
        x = x.reshape(-1, *x.shape[2:])

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))

        #(NT)1WH -> N(TW)H
        x = x.reshape(n, -1, x.size(-1))
        x = self.rnn(x)[0]
        x = self.classifier(x)
        x = x.reshape(
            x.size(0), 
            self.number_of_patches, 
            self.patch_size, 
            self.n_classes
        )
        return x