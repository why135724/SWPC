import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, in_chan=0, fc_num=0, out_chann=0):
        super(EEGNet, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, in_chan), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        self.fc1 = nn.Linear(fc_num, out_chann)

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(x.size()[0], -1)
        print('x',x.shape)
        x = self.fc1(x)
        return x


class EEGNet_feature(nn.Module):
    def __init__(self, in_chan=0,n_dim=None):
        super(EEGNet_feature, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, in_chan), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        self.fc = nn.Sequential(
            nn.Linear(n_dim, n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(n_dim, n_dim, bias=True),
        )

        self.dropout = nn.Dropout(p=0.25)  
        self.elu = nn.ELU()

    def forward(self, x, simsiam=False):
        # Layer 1
        x = self.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.dropout(x) #nn.Dropout(p = 0.3)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = self.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = self.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(x.size()[0], -1)
        # x = self.fc(x)
        # print('x',x.shape) #(64,376)
        # return x
        if simsiam:
            return x, self.fc(x)
        else:
            return x

class EEGNet_feature_wo_BN(nn.Module):
    def __init__(self, in_chan=0,n_dim=None):
        super(EEGNet_feature_wo_BN, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, in_chan), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        self.fc = nn.Sequential(
            nn.Linear(n_dim, n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(n_dim, n_dim, bias=True),
        )

        self.dropout = nn.Dropout(p=0.25)  
        self.elu = nn.ELU()

    def forward(self, x, simsiam=False):
        # Layer 1
        x = self.elu(self.conv1(x))
        # x = self.batchnorm1(x)
        x = self.dropout(x) #nn.Dropout(p = 0.3)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = self.elu(self.conv2(x))
        # x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = self.elu(self.conv3(x))
        # x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(x.size()[0], -1)
        # x = self.fc(x)
        print('x',x.shape) #(64,376)
        # return x
        if simsiam:
            return x, self.fc(x)
        else:
            return x


class EEGNet_class(nn.Module):
    def __init__(self, fc_num=0, out_chann=0):
        super(EEGNet_class, self).__init__()

        # FC Layer
        self.fc1 = nn.Linear(fc_num, out_chann)

    def forward(self, x):
        x = self.fc1(x)
        return x


class EEGNet_proj(nn.Module):
    def __init__(self,n_dim=None):
        super(EEGNet_proj, self).__init__()

        # FC Layer
        self.fc = nn.Sequential(
            nn.Linear(n_dim, n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(n_dim, n_dim, bias=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x