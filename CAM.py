import os 
import torch.nn as nn
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_img(root):
    img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])[:10]
    img_path_list = []
    for img_dir in img_dir_list:
        img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))[:3]]
        img_path_list.extend(img_path)
    return img_path_list


if __name__ == '__main__':

    methods = {
        "gradcam": GradCAM,
        "hirescam":HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }

    cuda_flag = True
    aug_smooth_flag = True
    eigen_smooth_flag = True
    method = "fullgrad"
    data_dir = "./data/CASIA_WebFace_20000"
    grp_list = [
        'eyes', 
        'mouth'
    ]
    for grp in grp_list:
        print(grp)
        # data_name = f'test_{grp}0.15_crop'
        data_name = 'test_crop'
        img_dir = os.path.join(data_dir, data_name)
        cam_dir = './result_cam_test/' + method + '_full'
        if not os.path.exists(cam_dir): 
            os.makedirs(cam_dir)
        epoch_list = [
            'baseModel',
            '15', '80'
        ]

        id_list = sorted(os.listdir(img_dir))
        label_dict = {id_list[i] : i for i in range(50)}

        for epoch in epoch_list:
            print(epoch)
            if grp == 'mouth':
                if epoch == 'baseModel':
                    log_dir = './logs/CASIA_WebFace_20000_0.15_test/[resnet50]_[0.01]_[0.98]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]'
                else:
                    log_dir = f'./logs/CASIA_WebFace_20000_0.15_test/[resnet50]_[0.01]_[0.98]_[train_crop]_[test_crop]_[mouth0.15_{epoch}]'
            else:
                if epoch == 'baseModel':
                    log_dir = './logs/CASIA_WebFace_20000_0.15_test/[resnet50]_[0.01]_[0.98]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]'
                else:
                    log_dir = f'./logs/CASIA_WebFace_20000_0.15_test/[resnet50]_[0.01]_[0.98]_[train_crop]_[test_crop]_[eyes0.15_{epoch}]'
            task = f'{data_name}_{grp}_{epoch}'

            # model
            model = models.resnet50()
            model.fc = nn.Linear(2048, 50)

            # model = models.vgg16()
            # model.classifier[6] = nn.Linear(4096, 50)

            para_dict = torch.load(os.path.join(log_dir, '150.pth'))
            model.load_state_dict(para_dict)

            # Choose the target layer you want to compute the visualization for.
            # Usually this will be the last convolutional layer in the model.
            # Some common choices can be:
            # Resnet18 and 50: model.layer4
            # VGG, densenet161: model.features[-1]
            # mnasnet1_0: model.layers[-1]
            # You can print the model to help chose the layer
            # You can pass a list with several target layers,
            # in that case the CAMs will be computed per layer and then aggregated.
            # You can also try selecting all layers of a certain type, with e.g:
            # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
            # find_layer_types_recursive(model, [torch.nn.ReLU])
            target_layers = [model.layer4]
            # target_layers = model.features[-1]

            img_path_list = get_img(img_dir)
            for img_path in img_path_list[:]:
                cls_name = img_path.split('/')[-2]
                img_name = img_path.split('/')[-1].split('.')[0]
                print('Now doing:', cls_name + '_' + img_name)
                rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
                rgb_img = np.float32(rgb_img) / 255
                input_tensor = preprocess_image(rgb_img)

                # We have to specify the target we want to generate
                # the Class Activation Maps for.
                # If targets is None, the highest scoring category (for every member in the batch) will be used.
                # You can target specific categories by
                # targets = [e.g ClassifierOutputTarget(281)]
                # cls = label_dict[cls_name]
                # targets = [ClassifierOutputTarget(cls)]
                targets = None

                # Using the with statement ensures the context is freed, and you can
                # recreate different CAM objects in a loop.
                cam_algorithm = methods[method]
                with cam_algorithm(model=model,
                                target_layers=target_layers,
                                use_cuda=cuda_flag) as cam:

                    # AblationCAM and ScoreCAM have batched implementations.
                    # You can override the internal batch size for faster computation.
                    cam.batch_size = 1
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        targets=targets,
                                        aug_smooth=aug_smooth_flag,
                                        eigen_smooth=eigen_smooth_flag)

                    # Here grayscale_cam has only one image in the batch
                    grayscale_cam = grayscale_cam[0, :]

                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=cuda_flag)
                # gb = gb_model(input_tensor, target_category=None)

                # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])s
                # cam_gb = deprocess_image(cam_mask * gb)
                # gb = deprocess_image(gb)

                save_dir = os.path.join(cam_dir, cls_name + '_' + img_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                cv2.imwrite(os.path.join(save_dir, task + '_cam.jpg'), cam_image)
                # break
                # cv2.imwrite(os.path.join(save_dir, task + '_gb.jpg'), gb)
                # cv2.imwrite(os.path.join(save_dir, task + '_cam_gb.jpg'), cam_gb)