import torch
from scipy import misc
from torch.autograd import Variable
import numpy as np
def total_variation(masks,power=2,border_penalty=0.3):
    #todo
    x_loss = torch.sum((torch.abs(masks[:,:,1:,:] - masks[:,:,:-1,:]))**power)
    y_loss = torch.sum((torch.abs(masks[:,:,:,1:] - masks[:,:,:-1,:]))**power)

    if border_penalty > 0:
        border = float(border_penalty)*torch.sum(masks[:,:,-1,:]**power + masks[:,:,0,:]**power + masks[:,:,:,-1]**power + masks[:,:,:,0]**power)
    else:
        border = 0

    
    return (x_loss + y_loss + border) / float(power*masks.size(0)) # normalized for batch size
def tensor_of_shape(x):
    return torch.zeros(*x.size())
def average_mask_loss(masks,power=0.3):
    if(power != 1):
        masks = (masks + 0.0005)**power # prevent nan derivative of sqrt at 0 is inf
    
    return torch.mean(masks)
def theta(images,masks):
    color_range = 0.66
    images = images.clone()
    n,c,_,_ = images.size()
    alt = tensor_of_shape(images) # gets tensor of zeros

    #color range
    alt += torch.Tensor(n,c,1,1).uniform_(-color_range/2,color_range/2)

    alt = Variable(alt, requires_grad=False)

    return (masks*images.detach()) + (1. - masks)*alt.detach()

def class_selector_loss(logits,labels):
    this = torch.sum(logits*labels,1)
    return torch.mean(this)

def one_hot(targets,dim=10):
   return Variable(torch.zeros( targets.size(0),dim).scatter_(1,targets.long().view(-1,1).data,1 ) )
    
def classifier_loss(images,masks,labels,black_box_func,lambda1=0.5,lambda2=8,lambda3=0.3,lambda4=0.2):
    
    labels = one_hot(labels)
    preserver_images = theta(images,masks)
    destroyer_images = theta(images,1 - masks)

    _, _, _, _, _,preserved_logits = black_box_func(preserver_images)
    _, _, _, _, _,destroyer_logits = black_box_func(destroyer_images)

    preserver_loss = torch.log(class_selector_loss( preserved_logits, labels ))
    destroyer_loss = class_selector_loss( destroyer_logits, labels ) 
    area_loss = average_mask_loss(masks)
    smoothness_loss = total_variation(masks)

    return lambda1*smoothness_loss + lambda2*area_loss - preserver_loss + lambda3*torch.pow(destroyer_loss,lambda4)

def test():
    from PIL import Image
    import numpy as np
    import os,pycat
    im = Variable(torch.Tensor(np.expand_dims(np.transpose(np.array(Image.open(os.path.join(os.path.dirname(__file__), 'test3.jpg'))), (2, 0, 1)), 0)/255.*2-1.), requires_grad=False)
    print('Original')
    misc.imshow(im[0].data.numpy())
    for pres in [1., 0.5, 0.1]:
        print('Mask strength =', pres)
        for e in range(5):
            m = Variable(torch.Tensor(1, 3, im.size(2), im.size(3)).fill_(pres), requires_grad=True)
            res = theta(im, m)
            misc.imshow(res[0].data.numpy())
    s = torch.sum(res)
    s.backward()
    print(torch.sum(m.grad))



'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f