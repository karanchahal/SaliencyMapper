import torch 
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import math


class SaliencyClassifier(nn.Module):

    def __init__(self,convs,class_size,batch_size):
        super(SaliencyClassifier,self).__init__()
        self.class_size = class_size
        self.batch_size = batch_size
        self.convs = convs
        self.classifier = nn.Conv2d(1024,class_size, kernel_size=3, padding=1)
        self.mask = Sequential([
            nn.Conv2d(2,size,kernel_size=1,padding=0),
            self.createMask()
        ])

        self._initialize_weights()
    
    def createMask(features):
        return abs(features[0]) / ( abs(features[0]) + abs(features[1]) )

    def forward(self,x):

        img = x

        scale1 = self.resnetScale1(x)
        scale2 = self.resnetScale2(x)
        scale3 = self.resnetScale3(x)
        scale4 = self.resnetScale4(x)
        scale5 = self.resnetScale5(x)

        features = this.feature_filter(scale5)
        upSampled = self.upSample(features)
        features = torch.cat([upSampled,scale4])

        upSampled = self.upSample(features)
        features = torch.cat([upSampled,scale3])

        upSampled = self.upSample(features)
        features = torch.cat([upSampled,scale2])

        x = self.mask(features)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

