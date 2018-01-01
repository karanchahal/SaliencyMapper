import torch 
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import math


class SaliencyClassifier(nn.Module):

    def __init__(self,class_size,batch_size):
        super(SaliencyClassifier,self).__init__()
        self.inplanes = 3
        self.class_size = class_size
        self.batch_size = batch_size
        self.resnetScale1 = self._make_layer(num_filters=[64],filter_sizes=[3],pool_size=2)
        self.resnetScale2 = self._make_layer(num_filters=[64],filter_sizes=[3],pool_size=2)
        self.resnetScale3 = self._make_layer(num_filters=[64],filter_sizes=[3],pool_size=8)
        
        self._initialize_weights()
    


    def _make_layer(self,num_filters,filter_sizes,pool_size):
        
        layers = []

        for i,filter_size in enumerate(filter_sizes):
            num_filter = num_filters[i]
            print(self.inplanes,num_filter)
            layers.append(
                nn.Conv2d(self.inplanes, num_filter,
                          kernel_size=filter_size, bias=False)
            )
            self.inplanes = num_filter
        
        layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=2, padding=1))

        return nn.Sequential(*layers)

    def forward(self,x):

        img = x

        scale1 = self.resnetScale1(x)
        print(scale1.size())

        scale2 = self.resnetScale2(scale1)
        print(scale2.size())

        scale3 = self.resnetScale3(scale2)
        print(scale3.size())

        # scale2 = self.resnetScale2(x)
        # scale3 = self.resnetScale3(x)
        # scale4 = self.resnetScale4(x)
        # scale5 = self.resnetScale5(x)

        # features = self.feature_filter(scale5)
        # upSampled = self.upSample(features)
        # features = torch.cat([upSampled,scale4])

        # upSampled = self.upSample(features)
        # features = torch.cat([upSampled,scale3])

        # upSampled = self.upSample(features)
        # features = torch.cat([upSampled,scale2])

        # x = self.mask(features)

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

x = torch.autograd.Variable(torch.randn((1,3,32,32)))
model = SaliencyClassifier(10,20)
o = model(x)