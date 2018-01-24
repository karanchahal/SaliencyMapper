import torch
from torch.autograd import Variable
from torch.nn import Module
from torch.nn.functional import conv2d
import torch.nn.functional as F
import numpy as np



class Loss: 
    
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.area_loss_coef = 8
        self.smoothness_loss_coef = 0.5
        self.preserver_loss_coef = 0.3
        self.area_loss_power = 0.3
    
    def get(self, masks, images, targets, black_box_func):
    
        one_hot_targets = self.one_hot(targets)
        
        area_loss = self.area_loss(masks)
        smoothness_loss = self.smoothness_loss(masks)
        destroyer_loss = self.destroyer_loss(images,masks,one_hot_targets,black_box_func)
        preserver_loss = self.preserver_loss(images,masks,one_hot_targets,black_box_func)
        
        
        return destroyer_loss + self.area_loss_coef*area_loss + self.smoothness_loss_coef*smoothness_loss + self.preserver_loss_coef*preserver_loss
        
    def one_hot(self,targets):
        depth = self.num_classes
        if targets.is_cuda:
            return Variable(torch.zeros(targets.size(0), depth).cuda().scatter_(1, targets.long().view(-1, 1).data, 1))
        else:
            return Variable(torch.zeros(targets.size(0), depth).scatter_(1, targets.long().view(-1, 1).data, 1))

  
    def tensor_like(self,x):
        if x.is_cuda:
            return torch.Tensor(*x.size()).cuda()
        else:
            return torch.Tensor(*x.size())
  
    def area_loss(self, masks):
        if self.area_loss_power != 1:
            masks = (masks+0.0005)**self.area_loss_power # prevent nan (derivative of sqrt at 0 is inf)

        return torch.mean(masks)
  
    def smoothness_loss(self,masks, power=2, border_penalty=0.3):
        x_loss = torch.sum((torch.abs(masks[:,:,1:,:] - masks[:,:,:-1,:]))**power)
        y_loss = torch.sum((torch.abs(masks[:,:,:,1:] - masks[:,:,:,:-1]))**power)
        if border_penalty>0:
            border = float(border_penalty)*torch.sum(masks[:,:,-1,:]**power + masks[:,:,0,:]**power + masks[:,:,:,-1]**power + masks[:,:,:,0]**power)
        else:
            border = 0.
        return (x_loss + y_loss + border) / float(power * masks.size(0))  # watch out, normalised by the batch size!
  
    def destroyer_loss(self,images,masks,targets,black_box_func):
        destroyed_images = self.apply_mask(images,1 - masks)
        out = black_box_func(destroyed_images)
        
        return self.cw_loss(out, targets, targeted=False, t_conf=1., nt_conf=5)
  
    def preserver_loss(self,images,masks,targets,black_box_func):
        preserved_images = self.apply_mask(images,masks)
        out = black_box_func(preserved_images)
        
        return self.cw_loss(out, targets, targeted=True, t_conf=1., nt_conf=1)
  
    def apply_mask(self,images, mask, noise=True, random_colors=True, blurred_version_prob=0.5, noise_std=0.11,
                 color_range=0.66, blur_kernel_size=55, blur_sigma=11,
                 bypass=0., boolean=False, preserved_imgs_noise_std=0.03):
        images = images.clone()
        cuda = images.is_cuda

        if boolean:
            # remember its just for validation!
            return (mask > 0.5).float() *images

        assert 0. <= bypass < 0.9
        n, c, _, _ = images.size()

        if preserved_imgs_noise_std > 0:
            images = images + Variable(self.tensor_like(images).normal_(std=preserved_imgs_noise_std) , requires_grad=False)
        if bypass > 0:
            mask = (1.-bypass)*mask + bypass
        if noise and noise_std:
            alt = self.tensor_like(images).normal_(std=noise_std)
        else:
            alt = self.tensor_like(images).zero_()
        if random_colors:
            if cuda:
                alt += torch.Tensor(n, c, 1, 1).cuda().uniform_(-color_range/2., color_range/2.)
            else:
                alt += torch.Tensor(n, c, 1, 1).uniform_(-color_range/2., color_range/2.)

        alt = Variable(alt, requires_grad=False)

        if blurred_version_prob > 0.: # <- it can be a scalar between 0 and 1
            cand = self.gaussian_blur(images, kernel_size=blur_kernel_size, sigma=blur_sigma)
            if cuda:
                when = Variable((torch.Tensor(n, 1, 1, 1).cuda().uniform_(0., 1.) < blurred_version_prob).float(), requires_grad=False)
            else:
                when = Variable((torch.Tensor(n, 1, 1, 1).uniform_(0., 1.) < blurred_version_prob).float(), requires_grad=False)
            alt = alt*(1.-when) + cand*when

        return (mask*images.detach()) + (1. - mask)*alt.detach()

    def cw_loss(self,logits, one_hot_labels, targeted=True, t_conf=2, nt_conf=5):

        this = torch.sum(logits*one_hot_labels, 1)
        other_best, _ = torch.max(logits*(1.-one_hot_labels) - 12111*one_hot_labels, 1)   # subtracting 12111 from selected labels to make sure that they dont end up a maximum
        t = F.relu(other_best - this + t_conf)
        nt = F.relu(this - other_best + nt_conf)
        if isinstance(targeted, (bool, int)):
            return torch.mean(t) if targeted else torch.mean(nt)

    def gaussian_blur(self,_images, kernel_size=55, sigma=11):
        ''' Very fast, linear time gaussian blur, using separable convolution. Operates on batch of images [N, C, H, W].
        Returns blurred images of the same size. Kernel size must be odd.
        Increasing kernel size over 4*simga yields little improvement in quality. So kernel size = 4*sigma is a good choice.'''
        
        kernel_a, kernel_b = self._gaussian_kernels(kernel_size=kernel_size, sigma=sigma, chans=_images.size(1))
        kernel_a = torch.Tensor(kernel_a)
        kernel_b = torch.Tensor(kernel_b)
        if _images.is_cuda:
            kernel_a = kernel_a.cuda()
            kernel_b = kernel_b.cuda()
        _rows = conv2d(_images, Variable(kernel_a, requires_grad=False), groups=_images.size(1), padding=(int(kernel_size / 2), 0))
        return conv2d(_rows, Variable(kernel_b, requires_grad=False), groups=_images.size(1), padding=(0, int(kernel_size / 2)) )


    def _gaussian_kernels(self,kernel_size, sigma, chans):
        assert kernel_size % 2, 'Kernel size of the gaussian blur must be odd!'
        x = np.expand_dims(np.array(range(int(-kernel_size/2), int(-kernel_size/2)+kernel_size, 1)), 0)
        vals = np.exp(-np.square(x)/(2.*sigma**2))
        _kernel = np.reshape(vals / np.sum(vals), (1, 1, kernel_size, 1))
        kernel =  np.zeros((chans, 1, kernel_size, 1), dtype=np.float32) + _kernel
        return kernel, np.transpose(kernel, [0, 1, 3, 2])

   