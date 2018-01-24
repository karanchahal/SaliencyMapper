import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable

from scipy import misc
from model import saliency_model
from resnet import resnet

def save_checkpoint(state, filename='sal.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(net,optimizer,filename='small.pth.tar'):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return net,optimizer

def cifar10():
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    return trainloader,testloader,classes





net = saliency_model()
net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

black_box_func = resnet()
loss_func = Loss(num_classes=10)

for epoch in range(3):  # loop over the dataset multiple times
    
    running_loss = 0.0
    running_corrects = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        mask,out = net(inputs,labels)
       
#         _, preds = torch.max(out.data, 1)
#         loss = criterion(out,labels)
        loss = loss_func.get(mask,inputs,labels,black_box_func)
      
#         running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.data[0]
        if(i%100 == 0):
          print('Epoch = %f , Loss = %f '%(epoch+1 , running_loss/(4*(i+1))) )
       
        loss.backward()
        optimizer.step()
    
    save_checkpoint(net,'saliency_model.tar')