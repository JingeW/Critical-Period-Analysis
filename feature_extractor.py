import os
import torch
import numpy as np
import pickle as pkl
import torch.nn as nn
from PIL import Image
import scipy.io as io
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor


def get_img(img_dir):
    img_dir_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
    img_path_list = []
    for img_dir in img_dir_list:
        img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        img_path_list.extend(img_path)
    return img_path_list


def read_img(path, size=224):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    img = transform(img)
    img = img.view(1, 3, size, size)
    return img


def get_feat_avg(feat):
    c, h, w = feat.shape
    feat = feat.reshape(c, -1)
    feat_avg = np.mean(feat, axis=1)
    return feat_avg

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Base model
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]/150.pth'
# save_name = 'fullFace_baseModel'

# Eyes model
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]/150.pth'
# save_name = 'eyesFoveated_baseModel'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_15]/150.pth'
# save_name = 'fullOnEyes_epoch15'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_80]/150.pth'
# save_name = 'fullOnEyes_epoch80'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[eyes0.15_15]/150.pth'
# save_name = 'mouthOnEyes_epoch15'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[eyes0.15_80]/150.pth'
# save_name = 'mouthOnEyes_epoch80'

# Mouth model
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]/150.pth'
# save_name = 'mouthFoveated_baseModel'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_15]/150.pth'
# save_name = 'fullOnMouth_epoch15'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]/150.pth'
# save_name = 'fullOnMouth_epoch80'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_15]/150.pth'
# save_name = 'eyesOnMouth_epoch15'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_80]/150.pth'
# save_name = 'eyesOnMouth_epoch80'

# Forehead model
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_forehead0.15_crop]_[test_forehead0.15_crop]_[baseModel]/150.pth'
# save_name = 'foreheadFoveated_baseModel'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_forehead0.12_50_crop]_[test_forehead0.12_50_crop]_[baseModel]/150.pth'
# save_name = 'foreheadFoveated_50_baseModel'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[forehead0.12_15]/150.pth'
# save_name = 'fullOnForehead12_epoch15'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[forehead0.12_80]/150.pth'
# save_name = 'fullOnForehead12_epoch80'
# pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[forehead0.15_15]/150.pth'
# save_name = 'fullOnForehead15_epoch15'
pretrain_path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[forehead0.15_80]/150.pth'
save_name = 'fullOnForehead15_epoch80'

return_nodes = {
    'layer1.2.relu_2': 'layer1',
    'layer2.3.relu_2': 'layer2',
    'layer3.5.relu_2': 'layer3',
    'layer4.2.relu_2': 'layer4',
}

model = models.resnet50()
model.fc = nn.Linear(2048, 50)
model.load_state_dict(torch.load(pretrain_path))
model.to(device)
model.eval()

# img_dir = './data/CelebA_face_new_cropped'
# img_path_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
img_dir = './data/CelebA500_cropped'
img_path_list = get_img(img_dir)
# save_dir = './export/' + save_name
save_dir = './fm32/' + save_name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

feat1_list, feat2_list, feat3_list, feat4_list = [], [], [], []
for img_path in img_path_list[:]:
    img_name = img_path.split('/')[-1]
    img_class = img_path.split('/')[-2]
    print('Now doing:', img_class + '/' + img_name)
    img = read_img(img_path).to(device)

    featExtractor = create_feature_extractor(model, return_nodes=return_nodes)
    intermediate_outputs = featExtractor(img)
    feat1 = intermediate_outputs['layer1'].cpu().detach().numpy().squeeze()
    feat2 = intermediate_outputs['layer2'].cpu().detach().numpy().squeeze()
    feat3 = intermediate_outputs['layer3'].cpu().detach().numpy().squeeze()
    feat4 = intermediate_outputs['layer4'].cpu().detach().numpy().squeeze()

    # feat1_avg = get_feat_avg(feat1)
    # feat2_avg = get_feat_avg(feat2)
    # feat3_avg = get_feat_avg(feat3)
    # feat4_avg = get_feat_avg(feat4)

    # feat1_list.append(feat1_avg)
    # feat2_list.append(feat2_avg)
    # feat3_list.append(feat3_avg)
    # feat4_list.append(feat4_avg)

    feat1_list.append(feat1.flatten())
    feat2_list.append(feat2.flatten())
    feat3_list.append(feat3.flatten())
    feat4_list.append(feat4.flatten())

feat1_list = np.array(feat1_list)
feat2_list = np.array(feat2_list)
feat3_list = np.array(feat3_list)
feat4_list = np.array(feat4_list)

print(feat1_list.shape)
print(feat2_list.shape)
print(feat3_list.shape)
print(feat4_list.shape)

with open(save_dir + '/fullFM_layer1.pkl', 'wb') as f:
    pkl.dump(feat1_list, f, protocol=pkl.HIGHEST_PROTOCOL)
with open(save_dir + '/fullFM_layer2.pkl', 'wb') as f:
    pkl.dump(feat2_list, f, protocol=pkl.HIGHEST_PROTOCOL)
with open(save_dir + '/fullFM_layer3.pkl', 'wb') as f:
    pkl.dump(feat3_list, f, protocol=pkl.HIGHEST_PROTOCOL)
with open(save_dir + '/fullFM_layer4.pkl', 'wb') as f:
    pkl.dump(feat4_list, f, protocol=pkl.HIGHEST_PROTOCOL)

# io.savemat(save_dir + '/FM_layer1.mat', mdict={'FM': feat1_list})
# io.savemat(save_dir + '/FM_layer2.mat', mdict={'FM': feat2_list})
# io.savemat(save_dir + '/FM_layer3.mat', mdict={'FM': feat3_list})
# io.savemat(save_dir + '/FM_layer4.mat', mdict={'FM': feat4_list})


