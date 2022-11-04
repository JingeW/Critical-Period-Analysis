import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def read_img(path, size=224):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(img)
    img = img.view(1, 3, size, size)
    return img


def get_feature(model_name, paradict_path, img, layer):
    if model_name == 'resnet50':
        return_nodes = {
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4"
        }
        model = models.resnet50()
        model.fc = nn.Linear(2048, 50)
    elif model_name == 'vgg16':
        return_nodes = {
            "features.4": "conv1",
            "features.9": "conv2",
            "features.16": "conv3",
            "features.23": "conv4",
            "features.30": "conv5"
        }
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, 50)
    else:
        return_nodes = {
            "features.1": "conv1",
            "features.4": "conv2",
            "features.7": "conv3",
            "features.9": "conv4",
            "features.11": "conv5"
        }
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096, 50)
    model.load_state_dict(torch.load(paradict_path))
    model.to(device)
    model.eval()
    featExtractor = create_feature_extractor(model, return_nodes=return_nodes)
    intermediate_outputs = featExtractor(img)
    feat = intermediate_outputs[layer].cpu().detach().numpy().squeeze()
    return feat


if __name__ == '__main__':
    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # img_path = '/home/sda1/Jinge/Attention_analysis/data/CASIA_WebFace_5000/test_mouth0.2_crop/0000965/071.jpg'
    img_path = '/home/sda1/Jinge/Attention_analysis/data/CASIA_WebFace_20000/test_crop/0000102/321.jpg'
    img_name = img_path.split('/')[-1]
    print('Now doing:', img_name)
    img = read_img(img_path)
    img = img.to(device)

    # epoch_list = ['40', '80', '120']
    epoch_list = ['15', '40', '80', '120', '160']
    paradict_path = './logs/CASIA_WebFace_20000_0.2/[resnet50]_[0.01]_[0.5]_[train_mouth0.2_crop]_[test_mouth0.2_crop]_[baseModel]/epoch.pth'
    model_name = 'resnet50'  # vgg16, resnet50, alex
    layer = 'layer1'  # conv3, layer1

    in_path = paradict_path.replace('epoch', epoch_list[0])
    in_feat = get_feature(model_name, in_path, img, layer)
    in_feat_avg = np.mean(in_feat, axis=0)
    in_feat_norm = in_feat_avg / np.sum(in_feat_avg)

    out_path = paradict_path.replace('epoch', epoch_list[-1])
    out_feat = get_feature(model_name, out_path, img, layer)
    out_feat_avg = np.mean(out_feat, axis=0)
    out_feat_norm = out_feat_avg / np.sum(out_feat_avg)

    plt.matshow(in_feat_avg)
    plt.matshow(out_feat_avg)

    # plt.matshow(in_feat_avg - out_feat_avg)
    # plt.colorbar()
    plt.matshow(out_feat_norm - in_feat_norm)
    plt.colorbar()