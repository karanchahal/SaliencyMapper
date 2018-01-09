import torch
from scipy import misc
from torch.autograd import Variable
import numpy as np
def total_variation(masks,power=2,border_penalty=0.3):
    #todo
    x_loss = torch.sum((torch.abs(masks[:,:1:,:] - masks[:,:,:-1,:]))**power)
    y_loss = torch.sum((torch.abs(masks[:,:,:,1:] - masks[:,:,:-1,:]))**power)

    if border_penalty > 0:
        border = float(border_penalty)*torch.sum(masks[:,:,-1,:]**power + masks[:,:,0,:]**power + masks[:,:,:,-1]**power + masks[:,:,:,0]**power)
    else:
        border = 0

    
    return (x_loss + y_loss + border) / float(power*masks.size(0)) # normalized for batch size
def tensor_of_shape(x):
    return torch.zeros(*x.size())
def average_mask_loss(masks):
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

    #gaussian probability
    #todo

    alt = Variable(alt, requires_grad=False)

    return (masks*images.detach()) + (1. - masks)*alt.detach()

# def class_selector_loss(logits,targets):
    
def one_hot(targets,dim=10):
   return Variable(torch.zeros( targets.size(0),dim).scatter_(1,targets.long().view(-1,1).data,1 ) )
    
def classfier_loss(images,masks,targets,black_box_func):
    
    targets = one_hot(targets)
    preserver_images = theta(images,masks)
    destroyer_images = theta(images,1 - masks)

    preserved_logits = black_box_func(preserver_images)
    destroyer_logits = black_box_func(destroyer_images)

    preserver_loss = class_selector_loss( preserved_logits )
    destroyer_loss = class_selector_loss( destroyer_logits )
    area_loss = average_mask_loss(masks)
    smoothness_loss = total_variation(masks)


    return 1

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

a = Variable(torch.ones((2,1)))
a[0,0] = 1
a[1,0] = 4
print(one_hot(a,10))