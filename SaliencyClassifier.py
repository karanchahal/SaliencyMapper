import torch 
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import math
import utils

class PixelShuffleBlock(nn.Module):
    def forward(self,x):
        return F.conv_transpose2d(x,filters,padding=1)
        return F.pixel_shuffle(x,2)
    

def UpSampleBlock(in_channels,out_channels,kernel_size=3):
    layers = [
        BottleneckBlock(inplanes=in_channels,num_filters=[out_channels*4],filter_sizes=[kernel_size])[0],
        nn.ConvTranspose2d(in_channels=out_channels*4,out_channels=out_channels*2,kernel_size=3,padding=0),
        nn.ConvTranspose2d(in_channels=out_channels*2,out_channels=out_channels,kernel_size=3,padding=0),
        nn.ReLU()
    ]
    
    return nn.Sequential(*layers)
    

def BottleneckBlock(num_filters,inplanes,filter_sizes,bn=True,activation=True,_list=False):
    
    layers = []

    for i,filter_size in enumerate(filter_sizes):
        num_filter = num_filters[i]
        layers.append(
            nn.Conv2d(inplanes, num_filter,
                        kernel_size=filter_size, bias=False)
        )
        if bn:
            layers.append(nn.BatchNorm2d(num_filter))
        if activation:
            layers.append(nn.ReLU())

        inplanes = num_filter
    
    # layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=2, padding=1))
    if _list:
        return layers

    return nn.Sequential(*layers),inplanes
    

class SaliencyClassifier(nn.Module):

    def __init__(self,class_size,batch_size):
        super(SaliencyClassifier,self).__init__()
        self.inplanes = 3
        BASE = 24

        self.class_size = class_size
        self.batch_size = batch_size

        self.scale0,self.inplanes = BottleneckBlock(num_filters=[BASE],inplanes=self.inplanes,filter_sizes=[3])
        self.scale1,self.inplanes = BottleneckBlock(num_filters=[BASE*2],inplanes=self.inplanes,filter_sizes=[3])
        self.scale2,self.inplanes = BottleneckBlock(num_filters=[BASE*4],inplanes=self.inplanes,filter_sizes=[3])
        self.scale3,self.inplanes = BottleneckBlock(num_filters=[BASE*8],inplanes=self.inplanes,filter_sizes=[3])

        self.scaleX = nn.AvgPool2d(kernel_size=16)
        self.fc = nn.Linear(BASE*8,class_size)

        self._initialize_weights()

    def forward(self,x):

        img = x
        scale0 = self.scale0(x)
        # print(scale0.size())

        scale1 = self.scale1(scale0)
        # print(scale1.size())

        scale2 = self.scale2(scale1)
        # print(scale2.size())

        scale3 = self.scale3(scale2)
        # print(scale3.size())

        scaleX = self.scaleX(scale3)
        scaleX = scaleX.view(self.batch_size,-1)

        # print(scaleX.size())

        scaleC = self.fc(scaleX)
        # print(scaleC.size())

        return scale0,scale1,scale2,scale3,scaleX,scaleC
    
    

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
    def __init__(self,class_size,batch_size):
        super(SaliencyModel,self).__init__()
        self.class_size = class_size
        self.batch_size = batch_size
        self.classifier = SaliencyClassifier(self.class_size,self.batch_size)
        self.upsample0 = UpSampler(in_channels=192,out_channels=96,passthrough_channels=96)
        self.upsample1 = UpSampler(in_channels=96,out_channels=48,passthrough_channels=48)
        self.upsample2 = UpSampler(in_channels=48,out_channels=24,passthrough_channels=24)
    
    def forward(self,x):
        s0,s1,s2,s3,sX,sC = self.classifier(x)
        print(s3.size())
        s2 = self.upsample0(s3,s2)
        print(s1.size())
        s1 = self.upsample1(s2,s1)
        print(s1.size())
        s0 = self.upsample2(s1,s0)
        print(s0.size())
        return s0


class UpSampler(nn.Module):
    def __init__(self,in_channels,out_channels,passthrough_channels):
        super(UpSampler, self).__init__()
        self.upsampler = UpSampleBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=3)
        bottleneck_in_channels = passthrough_channels + out_channels
        self.bottleneck,self.inplanes = BottleneckBlock(inplanes=bottleneck_in_channels,num_filters=[out_channels],filter_sizes=[1])

    def forward(self,x,passthrough):

        upsampled = self.upsampler(x)
        upsampled = torch.cat((upsampled,passthrough),1)
        return self.bottleneck(upsampled)

# class SaliencyLoss():
    
#     def__init__(self,classifier):
#         self.classifier = classifier
    
#     def get_loss(self,masks,images,targets):
        
        

# x = torch.autograd.Variable(torch.randn((1,3,32,32)))
# model = SaliencyModel(10,1)
# o = model(x)

