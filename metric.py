import os
import cv2
import torch
import numpy as np
import pickle as pkl
import torch.nn as nn
from utils_metric import *
from torchvision import models


# configs
data_dir = './data/CASIA_WebFace_20000'
data_name = 'test_crop'
img_dir = os.path.join(data_dir, data_name)
img_path_list = get_img(img_dir)
id_list = sorted(os.listdir(img_dir))
label_dict = {id_list[i] : i for i in range(50)}
stats_dir = './stats_final/' + 'gradcam'
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
cam_dir = './result_cam_final/' + 'gradcam'
if not os.path.exists(cam_dir):
    os.makedirs(cam_dir)

draw = True
combos = [
    ['baseModel', 'full', ''], ['baseModel', 'eyes', ''], ['baseModel', 'mouth', ''],
    ['full', 'eyes', '15'], ['full', 'eyes', '80'],
    ['full', 'mouth', '15'], ['full', 'mouth', '80'],
    ['region', 'eyes', '15'], ['region', 'eyes', '80'],
    ['region', 'mouth', '15'], ['region', 'mouth', '80'],
]
for combo in combos[:1]:
    target = combo[0]   # 'baseModel' | 'full', 'region'
    grp = combo[1]      # 'full', 'eyes', 'mouth' | 'eyes', 'mouth'
    epoch = combo[2]    # | '15', '80'

    log_root = './logs/CASIA_WebFace_20000_0.15_final/'
    arch = 'resnet50'
    lr = 1e-2
    gamma = 0.5
    bs = 32

    # locate log dir
    if target == 'baseModel':
        if grp == 'full':
            train_set, test_set = 'train_crop', 'test_crop'
        elif grp == 'eyes':
            train_set, test_set = 'train_eyes0.15_crop', 'test_eyes0.15_crop'
        else:
            train_set, test_set = 'train_mouth0.15_crop', 'test_mouth0.15_crop'
        log_dir = log_root + f'[{arch}]_[{lr}]_[{gamma}]_[{bs}]_[{train_set}]_[{test_set}]_[{target}]'
        task = f'{grp}_{target}'
    elif target == 'full':
        train_set, test_set = 'train_crop', 'test_crop'
        log_dir = log_root + f'[{arch}]_[{lr}]_[{gamma}]_[{bs}]_[{train_set}]_[{test_set}]_[{grp}0.15_{epoch}]'
        task = f'{target}_on_{grp}{epoch}'
    else:
        if grp == 'eyes':
            input = 'mouth'
            train_set, test_set = f'train_{input}0.15_crop', f'test_{input}0.15_crop'
        else:
            input = 'eyes'
            train_set, test_set = f'train_{input}0.15_crop', f'test_{input}0.15_crop'
        log_dir = log_root + f'[{arch}]_[{lr}]_[{gamma}]_[{bs}]_[{train_set}]_[{test_set}]_[{grp}0.15_{epoch}]'
        task = f'{input}_on_{grp}{epoch}'
    print(task)
    print(log_dir)

    # model
    model = models.resnet50()
    model.fc = nn.Linear(2048, 50)

    para_dict = torch.load(os.path.join(log_dir, '150.pth'))
    model.load_state_dict(para_dict)

    # declare the model used for grad-cam
    net = resnet(model)
    net.eval()

    # storage for res
    heatmap_list = []  
    metrics_list = []  # [Avg_eyes, Avg_mouth, Prop_eyes, Prop_mouth]
    eyesRegion_pixel_list = []
    mouthRegion_pixel_list = []
    # heatmap
    for img_path in img_path_list[:]:
        cls_name = img_path.split('/')[-2]
        img_name = img_path.split('/')[-1].split('.')[0]
        print('Now doing:', cls_name + '_' + img_name)

        img = read_img(img_path)
        pred = net(img)
        cls = label_dict[cls_name]
        pred[:, cls].backward()
        gradients = net.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = net.get_activations(img).detach()
        for i in range(len(pooled_gradients)): 
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        img = cv2.imread(img_path)
        heatmap = cv2.resize(np.array(heatmap), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_list.append(heatmap)

        if draw:
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap_color * 0.4 + img
            save_dir = os.path.join(cam_dir, cls_name + '_' + img_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(os.path.join(save_dir, task + '_map.jpg'), superimposed_img)

        # load landmarks
        landmark_path = img_path.replace('jpg', 'pkl').replace(data_name, data_name.split('_')[0] + '_landmark')
        with open(landmark_path, 'rb') as f:
            landmark_list = pkl.load(f)[0]
        fixes_eyes = [
            (landmark_list[0], landmark_list[5]),
            (landmark_list[1], landmark_list[6]),
            ((landmark_list[0] + landmark_list[1]) / 2, (landmark_list[5] + landmark_list[6]) / 2)
        ]
        fixes_mouth = [
            (landmark_list[3] , landmark_list[8]),
            (landmark_list[4], landmark_list[9]),
            ((landmark_list[3] + landmark_list[4]) / 2, (landmark_list[8] + landmark_list[9]) / 2 + abs(landmark_list[3] - landmark_list[4])/4)
        ]
        
        # load bounding box
        bbox_path = img_path.replace('jpg','pkl').replace(data_name, data_name.split('_')[0] + '_bbox')
        with open(bbox_path, 'rb') as f:
            bbox = pkl.load(f)

        img_ori_path = img_path.replace(data_name, data_name.split('_')[0])
        img_ori = cv2.imread(img_ori_path)
        mask_eyes = np.array(get_mask(img_ori, fixes_eyes, bbox))
        mask_mouth = np.array(get_mask(img_ori, fixes_mouth, bbox))
        eyesRegion_pixel_list.append(np.sum(mask_eyes))
        mouthRegion_pixel_list.append(np.sum(mask_mouth))
        # print(eyesRegion_pixel_list, mouthRegion_pixel_list)

        # draw contour of the metric region
        contour_path = os.path.join(save_dir, 'contour.jpg')
        if draw and not os.path.exists(contour_path):
            img_copy = img.copy()
            _, thresh_eyes = cv2.threshold(np.uint8(mask_eyes)*255, 125, 255, cv2.THRESH_BINARY)
            contours_eyes, _ = cv2.findContours(thresh_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            _, thresh_mouth = cv2.threshold(np.uint8(mask_mouth)*255, 125, 255, cv2.THRESH_BINARY)
            contours_mouth, _ = cv2.findContours(thresh_mouth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(img_copy, contours_eyes, -1, (0,0,255), 2)
            cv2.drawContours(img_copy, contours_mouth, -1, (0,0,255), 2)
            cv2.imwrite(contour_path, img_copy)

        # compute metric
        avg_eyes = zoneIntensityAvg(heatmap, mask_eyes)
        avg_mouth = zoneIntensityAvg(heatmap, mask_mouth)
        prop_eyes = zoneIntenstityRatio(heatmap, mask_eyes)
        prop_mouth = zoneIntenstityRatio(heatmap, mask_mouth)
        metrics_list.append([avg_eyes, avg_mouth, prop_eyes, prop_mouth])
        # print(metrics_list)

    # heatmap_array = np.array(heatmap_list)
    # heatmap_path = os.path.join(stats_dir, task + '_heatmaps.pkl')
    # with open(heatmap_path, 'wb') as f:
    #     pkl.dump(np.array(heatmap_array), f)

    # metrics_array = np.array(metrics_list)
    # metrics_path = os.path.join(stats_dir, task + '_metrics.pkl')
    # with open(metrics_path, 'wb') as f:
    #     pkl.dump(np.array(metrics_array), f)
    # print(metrics_array.shape)

    # print('saved!')
