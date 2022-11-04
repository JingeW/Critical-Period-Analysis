# '''
# TODO:
#     1. Get a img' featMap on certain layer
#     3. AVG cross channel to get avg-featMap of one layer
#     4. T-test between two avg-featMap
#     5. Visualize the difference between two avg-featmap
# '''

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torchvision.models as models


def get_feature(x, model, selected_layer):
    model = model.features

    for index, layer in enumerate(model):
        x = layer(x)
        if index == selected_layer:
            return x


def read_img(path, size=224):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(img)
    img = img.view(1, 3, size, size)
    return img


if __name__ == '__main__':
    # =====Configs=====
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img_path = '/home/sda1/Jinge/Attention_analysis/data/CASIA_WebFace_5000/test_crop/0004748/071.jpg'
    img_name = img_path.split('/')[-1]
    print('Now doing:', img_name)
    img = read_img(img_path)
    img = img.to(device)
    # features_layer_list = [4, 9, 16, 23, 30]

    in_path = './logs/CASIA_WebFace_5000_0.2/AttAnalysis_[vgg16]_[0.001]_[sgd]_[0.5]_[train_crop_mouth0.2_20]/100.pth'
    in_dict = torch.load(in_path)
    out_path = './logs/CASIA_WebFace_5000_0.2/AttAnalysis_[vgg16]_[0.001]_[sgd]_[0.5]_[train_crop_mouth0.2_80]/100.pth'
    out_dict = torch.load(out_path)

    in_model= models.vgg16()
    in_model.classifier[6] = nn.Linear(4096, 50)
    in_model.load_state_dict(in_dict)
    in_model.to(device)
    in_model.eval()
    with torch.no_grad():
        in_feature_out = get_feature(img, in_model, 16)
        in_feat = in_feature_out.cpu().detach().numpy().squeeze()
    in_feat_avg = np.mean(in_feat, axis=0)
    in_feat_norm = in_feat_avg / np.sum(in_feat_avg)

    out_model= models.vgg16()
    out_model.classifier[6] = nn.Linear(4096, 50)
    out_model.load_state_dict(out_dict)
    out_model.to(device)
    out_model.eval()
    with torch.no_grad():
        out_feature_out = get_feature(img, out_model, 16)
        out_feat = out_feature_out.cpu().detach().numpy().squeeze()
    out_feat_avg = np.mean(out_feat, axis=0)
    out_feat_norm = out_feat_avg / np.sum(out_feat_avg)

    print(in_feat_avg.shape, out_feat_avg.shape)

    plt.matshow(in_feat_avg)
    plt.matshow(out_feat_avg)
    plt.matshow(in_feat_avg - out_feat_avg)
    plt.colorbar()

    plt.figure()
    plt.matshow(in_feat_norm - out_feat_norm)
    plt.colorbar()
