from SaliencyClassifier import SaliencyModel, SaliencyClassifier

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import utils
from scipy import misc

def save_checkpoint(state, filename='sal.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(net,optimizer,filename='small.pth.tar'):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return net,optimizer
    
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

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



net = SaliencyModel(10,4)
black_box_func = SaliencyClassifier(10,4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer_bb = optim.SGD(black_box_func.parameters(), lr=0.0001, momentum=0.9)
black_box_func,_ = load_checkpoint(black_box_func,optimizer_bb)
net,optimizer = load_checkpoint(net,optimizer,'sal.pth.tar')
for epoch in range(10):  # loop over the dataset multiple times


    running_loss = 0.0
    running_corrects = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        masks = net(inputs,labels)

        loss = utils.classifier_loss(inputs,masks,labels,black_box_func)
        # print(loss.data[0])
        # misc.imshow(inputs[0].data.numpy())
        img = inputs[0].data.numpy().reshape((3,32,32))
        mask = masks[0].data.numpy().reshape((32,32))
        misc.imshow(img)
        misc.imshow(mask)

        loss.backward
        optimizer.step()
        # _, preds = torch.max(outputs.data, 1)
    break
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        
        # # print statistics
        # running_corrects += torch.sum(preds == labels.data)
        # running_loss += loss.data[0]
        # print(i)
        # print('Epoch = %f , Accuracy = %f, Loss = %f '%(epoch+1 , running_corrects/(4*(i+1)), running_loss/(4*(i+1))) )
    save_checkpoint({
        'state_dict': net.state_dict(),
        'optimizer' : optimizer.state_dict()
        })


# Testing 


dataiter = iter(testloader)
images, labels = dataiter.next()


# scale0,scale1,scale2,scale3,scaleX,outputs = net(Variable(images))
# _, predicted = torch.max(outputs.data, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

# correct = 0
# total = 0
# for data in testloader:
#     images, labels = data
#     outputs = net(Variable(images))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))

# print('Finished Training')