import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from scipy import misc
from model import saliency_model
from resnet import resnet
from loss import Loss

def save_checkpoint(state, filename='saliency_model.pth'):
    torch.save(state, filename)

#def load_checkpoint(net,optimizer,filename='small.pth.tar'):
 #   checkpoint = torch.load(filename)
  #  net.load_state_dict(checkpoint['state_dict'])
   # optimizer.load_state_dict(checkpoint['optimizer'])
    #return net,optimizer
#resnet() is better to load its whole model rather than its state_dict
def load_checkpoint(net,filename='./black_box_func.pth'):
    net = torch.load(filename)
    
 #when calculate the destroy loss and preserve loss ,the resnet should be pretrained , so we would load the black_box_func.pth   
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
from tqdm import tqdm

def train():
    num_epochs = 3
    trainloader,testloader,classes = cifar10()

    net = saliency_model()
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    black_box_func = resnet()
    black_box_func = black_box_func.cuda()
    black_box_func=load_checkpoint(black_box_func,filename='./black_box_func.pth')
    #load the pretrained classfication model 
    loss_func = Loss(num_classes=10)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        running_corrects = 0.0
        
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            mask,out = net(inputs,labels)
        
            loss = loss_func.get(mask,inputs,labels,black_box_func)
            running_loss += loss.data[0]

            if(i%10 == 0):
                print('Epoch = %f , Loss = %f '%(epoch+1 , running_loss/(4*(i+1))) )
        
            loss.backward()
            optimizer.step()
        
        save_checkpoint(net,'saliency_model.pth')

train()

