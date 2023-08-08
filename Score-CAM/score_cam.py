import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
from cam.scorecam import *
from PIL import Image


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


if __name__ == '__main__':
    # Configs
    data_dir = "./data/CASIA_WebFace_20000"
    grp_list = [
        'eyes', 
        'mouth'
    ]
    for grp in grp_list:
        print(grp)
        data_name = 'test_crop'
        img_dir = os.path.join(data_dir, data_name)
        cam_dir = './result_cam_final/' + 'socre_cam'
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir)
        epoch_list = ['baseModel', '15', '80']

        id_list = sorted(os.listdir(img_dir))
        label_dict = {id_list[i] : i for i in range(50)}

        for epoch in epoch_list:
            print(epoch)
            if grp == 'mouth':
                if epoch == 'baseModel':
                    log_dir = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]'
                else:
                    log_dir = f'./logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_{epoch}]'
            else:
                if epoch == 'baseModel':
                    log_dir = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]'
                else:
                    log_dir = f'./logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_{epoch}]'
            task = f'{data_name}_{grp}_{epoch}'

            # define model
            model = models.resnet50()
            model.fc = nn.Linear(2048, 50)

            # load state dict
            para_dict = torch.load(os.path.join(log_dir, '150.pth'))
            model.load_state_dict(para_dict)
            model.eval()

            # define cam
            model_dict = dict(type='resnet50', arch=model, layer_name='layer4',input_size=(224, 224))
            scorecam = ScoreCAM(model_dict)

            img_path_list = get_img(img_dir)
            for img_path in img_path_list[:]:
                cls_name = img_path.split('/')[-2]
                img_name = img_path.split('/')[-1].split('.')[0]
                print('Now doing:', cls_name + '_' + img_name)
                input_ = read_img(img_path)
                if torch.cuda.is_available():
                    input_ = input_.cuda()

                scorecam_map = scorecam(input_, class_idx=label_dict[cls_name])
                heatmap = scorecam_map.type(torch.FloatTensor).detach().cpu().squeeze()
                heatmap = np.maximum(heatmap, 0)
                heatmap /= torch.max(heatmap)

                img = cv2.imread(img_path)
                heatmap = cv2.resize(np.array(heatmap), (img.shape[1], img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 + img

                save_dir = os.path.join(cam_dir, cls_name + '_' + img_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, task + '_map.jpg'), superimposed_img)
                # break