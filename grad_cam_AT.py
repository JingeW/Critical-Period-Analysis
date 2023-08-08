import os
import cv2
import torch
import numpy as np
import pickle as pkl
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, models

class resnet(nn.Module):
    def __init__(self, pretrained_model):
        super(resnet, self).__init__()

        self.net = pretrained_model
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4
        self.avgpool = self.net.avgpool
        self.classifier = self.net.fc

    def activation_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        h = x.register_hook(self.activation_hook)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def get_img(root):
    # img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])[:10]
    img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])
    img_path_list = []
    for img_dir in img_dir_list:
        # img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))[:5]]
        img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        img_path_list.extend(img_path)
    return img_path_list


def read_img(path):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    img = transform(img)
    img = img.view(1, 3, 224, 224)
    return img

T = 3.5
alpha = 1
beta = 100

image_path = './data/CASIA_WebFace_20000/test_crop/0000102/321.jpg'
model_path = f'./logs/CASIA_WebFace_20000_0.15_final_recovery_v1/[resnet50]_[{T}]_[{alpha}]_[{beta}]_[mouth0.15_80]/150.pth'
# model_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]/150.pth'
# model_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]/150.pth'


model = models.resnet50()
model.fc = nn.Linear(2048, 50)
# para_dict = torch.load(model_path)
# model.load_state_dict(para_dict)
net = resnet(model)
para_dict = torch.load(model_path)
net.load_state_dict(para_dict)
net.eval()
cls_name = image_path.split('/')[-2]
img_name = image_path.split('/')[-1].split('.')[0]
print('Now doing:', cls_name + '_' + img_name)
# break
# get the most likely prediction of the model
img = read_img(image_path)
pred = net(img)
cls = int(pred.argmax(1))
# get the gradient of the output with respect to the parameters of the model
pred[:, cls].backward()
# pull the gradients out of the model
gradients = net.get_activations_gradient()
# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
# get the activations of the last convolutional layer
activations = net.get_activations(img).detach()
# weight the channels by corresponding gradients
for i in range(len(pooled_gradients)):
    activations[:, i, :, :] *= pooled_gradients[i]
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()
print(heatmap.max(), heatmap.min())
# relu on top of the heatmap
heatmap = np.maximum(heatmap, 0)
# normalize the heatmap
heatmap /= torch.sum(heatmap)
heatmap /= torch.max(heatmap)
# heatmap /= torch.sum(heatmap)
# draw the heatmap
# plt.matshow(heatmap.squeeze())
# interpolate the heat-map and project it onto the original image
img = cv2.imread(image_path)
heatmap = cv2.resize(np.array(heatmap), (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
# plt.matshow(superimposed_img)
cv2.imwrite('./' + str(T) + '_' + str(alpha)+ '_' +str(beta) + '_recovery_map.jpg', superimposed_img)
# cv2.imwrite('./' +  'mouth_base_map.jpg', superimposed_img)
