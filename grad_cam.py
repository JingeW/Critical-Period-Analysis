import os
import cv2
import torch
import numpy as np
import pickle as pkl
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, models
from model.vgg16_eca import VGG16_eca


class VGG_eca(nn.Module):
    def __init__(self, pretrained_model):
        super(VGG_eca, self).__init__()

        self.vgg_eca = pretrained_model

        self.conv1 =  self.vgg_eca.Conv1
        self.ECA1 =  self.vgg_eca.ECA1

        self.conv2 =  self.vgg_eca.Conv2
        self.ECA2 =  self.vgg_eca.ECA2

        self.conv3 =  self.vgg_eca.Conv3
        self.ECA3 =  self.vgg_eca.ECA3

        self.conv4 =  self.vgg_eca.Conv4
        self.ECA4 =  self.vgg_eca.ECA4

        self.conv5 =  self.vgg_eca.Conv5
        self.ECA5 =  self.vgg_eca.ECA5

        self.max_pool = self.vgg_eca.maxpool

        self.classifier = self.vgg_eca.classifier

        self.gradients = None

    def activation_hook(self, grad):
        self.gradients= grad

    def forward(self, x):
        x = self.conv1(x)
        A1, x = self.ECA1(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        A2, x = self.ECA2(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        A3, x = self.ECA3(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        A4, x = self.ECA4(x)
        x = self.max_pool(x)

        x = self.conv5(x)
        A5, x = self.ECA5(x)
        
        h = x.register_hook(self.activation_hook)

        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.conv1(x)
        A1, x = self.ECA1(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        A2, x = self.ECA2(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        A3, x = self.ECA3(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        A4, x = self.ECA4(x)
        x = self.max_pool(x)

        x = self.conv5(x)
        A5, x = self.ECA5(x)
        return x


class VGG(nn.Module):
    def __init__(self, pretrained_model):
        super(VGG, self).__init__()

        self.vgg = pretrained_model

        self.features_conv = self.vgg.features[:30] # vgg16: 30; alex: 12

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.classifier = self.vgg.classifier

        self.gradients = None


    def activation_hook(self, grad):
        self.gradients= grad

    def forward(self, x):
        x = self.features_conv(x)

        h = x.register_hook(self.activation_hook)

        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)


class resnet(nn.Module):
    def __init__(self, pretrained_model):
        super(resnet, self).__init__()

        self.resnet = pretrained_model
        
        self.conv1 = self.resnet.conv1

        self.bn1 = self.resnet.bn1

        self.relu = self.resnet.relu

        self.maxpool = self.resnet.maxpool

        self.layer1 = self.resnet.layer1

        self.layer2 = self.resnet.layer2

        self.layer3 = self.resnet.layer3

        self.features_conv = self.resnet.layer4

        self.avgpool = self.resnet.avgpool

        self.classifier = self.resnet.fc

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
        x = self.features_conv(x)

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
        return self.features_conv(x)


def get_img(root):
    # img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])[:10]
    img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])
    img_path_list = []
    for img_dir in img_dir_list:
        # img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))[:5]]
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

if __name__ == '__main__':
    epoch_list = [
        'baseModel',
        '15', '40', '80'
    ]
    for epoch in epoch_list:
        print(epoch)
        data_dir = "./data/CASIA_WebFace_20000"
        data_name = 'test_mouth0.15_crop'
        img_dir = os.path.join(data_dir, data_name)
        if epoch == 'baseModel':
            log_dir = './logs/CASIA_WebFace_20000_0.15/[vgg16]_[0.001]_[0.5]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]'
            # log_dir = './logs/CASIA_WebFace_20000_0.15/[resnet50]_[0.01]_[0.5]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]'
        else:
            log_dir = f'./logs/CASIA_WebFace_20000_0.15/[vgg16]_[0.001]_[0.5]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_{epoch}]'
            # log_dir = f'./logs/CASIA_WebFace_20000_0.15/[resnet50]_[0.01]_[0.5]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[eyes0.15_{epoch}]'
        # task = f'train_crop_mouth0.15_{epoch}'
        task = f'train_crop_mouth0.15_{epoch}'
        # cam_dir = './cam_result/' + task
        # if not os.path.exists(cam_dir):
        #     os.makedirs(cam_dir)

        # model
        # model = models.resnet50()
        # model.fc = nn.Linear(2048, 50)
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, 50)
        # model = VGG16_eca(50)

        # load state dict
        para_dict = torch.load(os.path.join(log_dir, '120.pth'))
        model.load_state_dict(para_dict)

        # declare the model used for grad-cam
        net = VGG(model)
        # net = resnet(model)
        net.eval()

        # grad-cam
        img_path_list = get_img(img_dir)
        for img_path in img_path_list[:]:
            cls_name = img_path.split('/')[-2]
            img_name = img_path.split('/')[-1].split('.')[0]
            print('Now doing:', cls_name + '_' + img_name)
            # break
            # get the most likely prediction of the model
            img = read_img(img_path)
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
            # relu on top of the heatmap
            heatmap = np.maximum(heatmap, 0)
            # normalize the heatmap
            heatmap /= torch.max(heatmap)
            # draw the heatmap
            # plt.matshow(heatmap.squeeze())
            # interpolate the heat-map and project it onto the original image
            img = cv2.imread(img_path)
            heatmap = cv2.resize(np.array(heatmap), (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img
            # plt.matshow(superimposed_img)
            cv2.imwrite('./' + task + '_map.jpg', superimposed_img)
            break


            # # load fixes and bbox
            # landmark_path = img_path.replace('jpg', 'pkl').replace(data_name, data_name.split('_')[0] + '_landmark')
            # with open(landmark_path, 'rb') as f:
            #     landmark_list = pkl.load(f)
            # landmark_list = landmark_list[0]
            # if region == 'eyes':
            #     fixes = [
            #         (landmark_list[0], landmark_list[5]),
            #         (landmark_list[1], landmark_list[6]),
            #         ((landmark_list[0] + landmark_list[1]) / 2, (landmark_list[5] + landmark_list[6]) / 2)
            #     ]
            # else:
            #     fixes = [
            #         (landmark_list[3] , landmark_list[8]),
            #         (landmark_list[4], landmark_list[9]),
            #         # ((landmark_list[3] + landmark_list[4]) / 2, (landmark_list[8] + landmark_list[9]) / 2)
            #         ((landmark_list[3] + landmark_list[4]) / 2, (landmark_list[8] + landmark_list[9]) / 2 + abs(landmark_list[3] - landmark_list[4])/4)

            #     ]
            # bbox_path = img_path.replace('jpg','pkl').replace(data_name, data_name.split('_')[0] + '_bbox')
            # with open(bbox_path, 'rb') as f:
            #     bbox = pkl.load(f)

            # # create mask for cropped input
            # img_ori_path = img_path.replace(data_name, data_name.split('_')[0])
            # img_ori = cv2.imread(img_ori_path)
            # mask = getMask(img_ori, fixes, bbox)

            # # draw contour of the metric region
            # img_copy = img.copy()
            # mask_cv = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
            # mask_gray = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY)
            # ret, thresh = cv2.threshold(mask_gray, 125, 255, cv2.THRESH_BINARY)
            # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img_copy, contours, -1, (0,0,255), 2)

            # # compute metric
            # heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
            # avg = zoneIntensityAvg(heatmap_gray, mask_gray)
            # prop = zoneIntenstityRatio(heatmap_gray, mask_gray)
            # # metrics = [avg, prop]

            # # save result
            # save_path = os.path.join(cam_dir, cls_name, img_name.split('_')[0])
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # cv2.imwrite(save_path + '/' + task + '_map.jpg', superimposed_img)
            # if task == 'eyes_new_crop' or task == 'mouth_new_crop':
            #     cv2.imwrite(save_path + '/' + region + '_contour.jpg', img_copy)
            # with open(save_path + '/' 'metrics.txt', 'a') as f:
            #     line = task + ' ' + region + ' ' + str(f'{avg: .3f}') + ' ' + str(f'{prop: .3f}')
            #     f.write(line)
            #     f.write('\n')
    
            # # save metric 
            # metric_dir = res_dir + 'metrics'
            # if not os.path.exists(metric_dir):
            #     os.makedirs(metric_dir)
            # with open(metric_dir + '/' + task + '_metrics.csv', 'a+') as f:
            #     # line = task + '_' + region + ': ' + str(metrics[0]) + ', ' + str(metrics[1])
            #     line = cls_name + '_' + img_name + ',' + str(f'{avg: .3f}') + ', ' + str(f'{prop: .3f}')
            #     f.write(line)
            #     f.write('\n')
    
