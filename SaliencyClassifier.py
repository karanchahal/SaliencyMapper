import torch 
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import math


class SaliencyClassifier(nn.Module):

    def __init__(self,class_size,batch_size):
        super(SaliencyClassifier,self).__init__()
        self.inplanes = 3
        BASE = 24
        self.class_size = class_size
        self.batch_size = batch_size
        self.scale0 = self._make_layer(num_filters=[BASE],filter_sizes=[3])
        self.scale1 = self._make_layer(num_filters=[BASE*2],filter_sizes=[3])
        self.scale2 = self._make_layer(num_filters=[BASE*4],filter_sizes=[3])
        self.scale3 = self._make_layer(num_filters=[BASE*8],filter_sizes=[3])
        self.scaleX = nn.AvgPool2d(kernel_size=16)
        self.fc = nn.Linear(BASE*8,class_size)
        self._initialize_weights()

    def forward(self,x):

        img = x
        scale0 = self.scale0(x)
        print(scale0.size())

        scale1 = self.scale1(scale0)
        print(scale1.size())

        scale2 = self.scale2(scale1)
        print(scale2.size())

        scale3 = self.scale3(scale2)
        print(scale3.size())

        scaleX = self.scaleX(scale3)
        scaleX = scaleX.view(self.batch_size,-1)

        print(scaleX.size())

        scaleC = self.fc(scaleX)
        print(scaleC.size())

        return scale0,scale1,scale2,scale3,scaleX,scaleC
    
    def _make_layer(self,num_filters,filter_sizes,bn=True,activation=True):
        
        layers = []

        for i,filter_size in enumerate(filter_sizes):
            num_filter = num_filters[i]
            layers.append(
                nn.Conv2d(self.inplanes, num_filter,
                          kernel_size=filter_size, bias=False)
            )
            if bn:
                layers.append(nn.BatchNorm2d(num_filter, affine=affine))
            if activation:
                layers.append(nn.ReLU())

            self.inplanes = num_filter
        
        # layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=2, padding=1))

        return nn.Sequential(*layers)

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


class SaliencyModel(nn.Module):
    def __init__(self,batch_size,class_size):
        self.class_size = class_size
        self.batch_size = batch_size

        self.classifier = SaliencyClassifier(self.class_size,self.batch_size)
        #fefe
    
    def forward(self,x):

class UpSampler(nn.Module):
    def __init__():
        self.upsampler = upsample_block(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=3)
        bottleneck_in_channels = passthrough_channels + out_channels
        self.bottleneck = _make_layer(num_filters=[bottleneck_in_channels],filter_size=[3])
    def forward(self,x,passthrough):
        
        upsampled = self.upsampler(x)
        upsampled = torch.cat((upsampled,passthrough),1)

        return self.bottleneck(upsampled)

x = torch.autograd.Variable(torch.randn((1,3,32,32)))
model = SaliencyClassifier(10,1)
o = model(x)