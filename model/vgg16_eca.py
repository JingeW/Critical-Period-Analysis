import torch.nn as nn
from math import log

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        self.channel = channel
        t = int(abs((log(self.channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)

        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return y, x * y.expand_as(x)


class VGG16_eca(nn.Module):
    def __init__(self, out_num):
        super(VGG16_eca, self).__init__()
        self.out_num = out_num

        # ======features======
        # Conv1
        block1 = []
        block1.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        block1.append(nn.ReLU(inplace=True))
        block1.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        block1.append(nn.ReLU(inplace=True))

        self.Conv1 = nn.Sequential(*block1)
        self.ECA1 = ECA(64)

        # Conv2
        block2 = []
        block2.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        block2.append(nn.ReLU(inplace=True))
        block2.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        block2.append(nn.ReLU(inplace=True))

        self.Conv2 = nn.Sequential(*block2)
        self.ECA2 = ECA(128)

        # Conv3
        block3 = []
        block3.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        block3.append(nn.ReLU(inplace=True))
        block3.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        block3.append(nn.ReLU(inplace=True))
        block3.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        block3.append(nn.ReLU(inplace=True))
        
        self.Conv3 = nn.Sequential(*block3)
        self.ECA3 = ECA(256)

        # Conv4
        block4 = []
        block4.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        block4.append(nn.ReLU(inplace=True))
        block4.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        block4.append(nn.ReLU(inplace=True))
        block4.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        block4.append(nn.ReLU(inplace=True))

        self.Conv4 = nn.Sequential(*block4)
        self.ECA4 = ECA(512)

        # Conv5
        block5 = []
        block5.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        block5.append(nn.ReLU(inplace=True))
        block5.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        block5.append(nn.ReLU(inplace=True))
        block5.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        block5.append(nn.ReLU(inplace=True))

        self.Conv5 = nn.Sequential(*block5)
        self.ECA5 = ECA(512)

        # Maxpool
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ======Classifier======
        fc = []
        # FC6
        fc.append(nn.Linear(in_features=512 * 7 * 7, out_features=4096))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout(p=0.5))
        # FC7
        fc.append(nn.Linear(in_features=4096, out_features=4096))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout(p=0.5))
        # FC8 output
        fc.append(nn.Linear(in_features=4096, out_features=self.out_num))

        self.classifier = nn.Sequential(*fc)

        # ======Initiation======
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv1 = self.Conv1(x)
        A1, conv1 = self.ECA1(conv1)
        conv1 = self.maxpool(conv1) # out_dim: 112, 112, 64

        conv2 = self.Conv2(conv1)
        A2, conv2 = self.ECA2(conv2)
        conv2 = self.maxpool(conv2) # out_dim: 56, 56, 128

        conv3 = self.Conv3(conv2)
        A3, conv3 = self.ECA3(conv3)
        conv3 = self.maxpool(conv3) # out_dim: 28, 28, 256

        conv4 = self.Conv4(conv3)
        A4, conv4 = self.ECA4(conv4)
        conv4 = self.maxpool(conv4) # out_dim: 14, 14, 512

        conv5 = self.Conv5(conv4)
        A5, conv5 = self.ECA5(conv5)
        conv5 = self.maxpool(conv5) # out_dim: 7, 7, 512

        conv5_f = conv5.view(conv5.size(0), -1)
        out = self.classifier(conv5_f)

        return out

# model = VGG16_eca(50)
# print(model)