import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


def get_img_new(data_dir):
    img_path_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)], key=str.casefold)
    return img_path_list


def read_img(path, size=224):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(img)
    img = img.view(1, 3, size, size)
    return img


def get_feature(paradict_path, img):
    return_nodes = {
        "layer1": "layer1",
        "layer2": "layer2",
        "layer3": "layer3",
        "layer4": "layer4"
    }
    model = models.resnet50()
    model.fc = nn.Linear(2048, 50)
    model.load_state_dict(torch.load(paradict_path))
    model.to(device)
    model.eval()
    featExtractor = create_feature_extractor(model, return_nodes=return_nodes)
    intermediate_outputs = featExtractor(img)
    feats = [intermediate_outputs[layer].cpu().detach().numpy().squeeze() for layer in intermediate_outputs.keys()]
    return feats


if __name__ == '__main__':
    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_dir = './data/'
    grp_list = [
            'eyes',
            'mouth'
        ]
    for grp in grp_list:
        print(grp)
        data_name = 'CelebA_face_new_cropped'
        img_dir = os.path.join(data_dir, data_name)
        img_path_list = get_img_new(img_dir)
        fm_dir = './featmap/' + data_name
        if not os.path.exists(fm_dir):
            os.makedirs(fm_dir)
        epoch_list = ['baseModel', '15', '80']

        id_list = sorted(os.listdir(img_dir))
        label_dict = {id_list[i] : i for i in range(50)}

        for epoch in epoch_list:
            print(epoch)
            if grp == 'mouth':
                if epoch == 'baseModel':
                    log_dir = './logs/CASIA_WebFace_20000_0.15/[resnet50]_[0.01]_[0.5]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]'
                else:
                    log_dir = f'./logs/CASIA_WebFace_20000_0.15/[resnet50]_[0.01]_[0.5]_[train_crop]_[test_crop]_[mouth0.15_{epoch}]'
            else:
                if epoch == 'baseModel':
                    log_dir = './logs/CASIA_WebFace_20000_0.15/[resnet50]_[0.01]_[0.5]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]'
                else:
                    log_dir = f'./logs/CASIA_WebFace_20000_0.15/[resnet50]_[0.01]_[0.5]_[train_crop]_[test_crop]_[eyes0.15_{epoch}]'
            task = f'{data_name}_{grp}_{epoch}'

            paradict_path = torch.load(os.path.join(log_dir, '150.pth'))
            features_layer1 = features_layer2 = features_layer3 = features_layer4 = []
            for img_path in img_path_list:
                img = read_img(img_path)
                img.to(device)
                feats = get_feature(paradict_path, img)
                features_layer1.append(feats[0])
                features_layer1.append(feats[1])
                features_layer1.append(feats[2])
                features_layer1.append(feats[3])

