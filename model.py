import resnet
from scipy import misc
import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from PIL import Image

model = resnet.resnet50()


imsize = 256
loader = transforms.Compose([
     transforms.Scale(imsize),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
     ])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image #assumes that you're using GPU

image = image_loader('./test.jpg')

out = model(image)
#print(out)
maxk = 3
_, pred = out.topk(maxk, 1, True, True)
pred = pred.t()

print(pred)
correct = pred.eq(target.view(1, -1).expand_as(pred))

print(correct)
_, preds = torch.max(out.data, 1)
# preds = preds.numpy()

# print(classes[preds[0]])
# # print(preds)