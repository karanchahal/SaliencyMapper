import resnet


model = resnet.resnet50(pretrained=True)
print(model)