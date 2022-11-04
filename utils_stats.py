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

        self.gradients = None


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
    # img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])[:3]
    img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])
    img_path_list = []
    for img_dir in img_dir_list:
        # img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))[:3]]
        img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        img_path_list.extend(img_path)
    return img_path_list


def get_img_new(data_dir):
    img_path_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)], key=str.casefold)
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


def get_mask(img_ori, fixes, bbox):
    height, width, _ = img_ori.shape

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixes[0][0]) ** 2 + (y2d - fixes[0][1]) ** 2)
    for fix in fixes[1:]:
        theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2))

    th =  np.sqrt((fixes[0][0] - fixes[2][0]) ** 2 + (fixes[0][1] - fixes[2][1]) ** 2)

    mask = np.zeros_like(theta, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            if theta[i][j] <= th:
                mask[i][j] = 1
            else:
                mask[i][j] = 0

    bbox = bbox.squeeze()
    # mask = Image.fromarray(np.uint8(mask)*255)
    mask = Image.fromarray(mask)
    mask_crop = mask.crop(
        (bbox[0], bbox[1], bbox[2], bbox[3],)
    )

    return mask_crop


def zoneIntensityAvg(heatmap, mask):
    zone = heatmap * mask
    zone_total = sum(map(sum, zone))
    mask_total = sum(map(sum, mask))
    zone_avg = zone_total / mask_total

    return zone_avg


def zoneIntenstityRatio(heatmap, mask):
    zone = heatmap * mask
    zone_total = sum(map(sum, zone))
    map_total = sum(map(sum, heatmap))
    zone_ratio = zone_total/map_total

    return zone_ratio
