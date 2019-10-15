import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from scipy import misc
from resnet import resnet

def save_checkpoint(state, filename='black_box_func.pth'):
    torch.save(state, filename)
#it seems wrong tar ,so convert to pth
#def load_checkpoint(net,optimizer,filename='black_box_func.pth'):
 #   checkpoint = torch.load(filename)
  #  net.load_state_dict(checkpoint['state_dict'])
   # optimizer.load_state_dict(checkpoint['optimizer'])
    #return net,optimizer
#it has not been used for the code

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
 
trainloader,testloader,classes = cifar10()

black_box_func = resnet()
black_box_func = black_box_func.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(black_box_func.parameters())

for epoch in range(6):  # loop over the dataset multiple times

    
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
        out = black_box_func(inputs)

        _, preds = torch.max(out.data, 1)
        loss = criterion(out,labels)   
        running_corrects += torch.sum(preds == labels.data)
        running_corrects=running_corrects.float()#avoid the acc=0
        running_loss += loss.data
        if(i%100 == 0):
          print('Epoch = %f , Accuracy = %f, Loss = %f '%(epoch+1 , running_corrects/(4*(i+1)), running_loss/(4*(i+1))) )
       
        loss.backward()
        optimizer.step()
    
    save_checkpoint(black_box_func, filename='black_box_func.pth')
