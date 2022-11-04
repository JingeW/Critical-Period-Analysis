import os
import copy
import torch
import numpy as np
import torch.nn as nn
import pickle as pkl
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.utils import save_image
from model.vgg16_eca import VGG16_eca


def toconv(layers):
    newlayers = []
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            newlayer = None
            if i == 0:
                m, n = 512, layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 7, 7))
            else:
                m, n = layer.weight.shape[1], layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers


def toconv_new(layers):
    newlayers = []
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            newlayer = None
            if i == 1:
                m, n = 256, layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 6)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 6, 6))
            else:
                m, n = layer.weight.shape[1], layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers


def newlayer(layer, g):
    layer = copy.deepcopy(layer)
    try:
        layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError:
        pass
    try:
        layer.bias = nn.Parameter(g(layer.bias))
    except AttributeError:
        pass
    return layer


def heatmap(R, sx, sy):
    b = 10 * ((np.abs(R)**3.0).mean()**(1.0 / 3))
    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    plt.show()


def digit(X, sx, sy):
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.imshow(X, interpolation='nearest', cmap='gray')
    plt.show()


def image(X, sx, sy):
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.imshow(X, interpolation='nearest')
    plt.show()


def get_img(root):
    # img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])[:10]
    img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])
    img_path_list = []
    for img_dir in img_dir_list:
        # img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))[:5]]
        img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        img_path_list.extend(img_path)
    return img_path_list


def get_img_new(root):
    img_path_list = sorted([os.path.join(root, f) for f in os.listdir(root)])
    return img_path_list


def read_img(path):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(img)
    # print(img.size())
    img = img.view(1, 3, 224, 224)
    return img


if __name__ == '__main__':
    data_dir = "./data/CASIA_WebFace_5000"
    data_name = 'test_crop'
    epoch = '120'
    img_dir = os.path.join(data_dir, data_name)
    log_dir = f'./logs/CASIA_WebFace_5000_0.2/[alex]_[0.001]_[0.5]_[train_crop_mouth0.2_{epoch}]'
    task = f'train_crop_mouth0.2_{epoch}'
    lrp_dir = './lrp_result/' + task
    if not os.path.exists(lrp_dir):
        os.makedirs(lrp_dir)

    imageClass = sorted(os.listdir(img_dir))
    num_class = 50

    # model
    model = models.alexnet()
    model.classifier[6] = nn.Linear(4096, 50)

    # model = VGG16_eca(50)
    
    # load the new state dict
    para_dict = torch.load(os.path.join(log_dir, '150.pth'))
    model.load_state_dict(para_dict)
    model.eval()

    # LRP
    layers = list(model._modules['features']) + toconv_new(list(model._modules['classifier']))
    L = len(layers)

    R_dict = {}
    img_path_list = get_img(img_dir)
    for img_path in img_path_list[1234:]:
        cls_name = img_path.split('/')[-2]
        img_name = img_path.split('/')[-1].split('.')[0]
        print('Now doing:', cls_name + '_' + img_name)

        img = read_img(img_path)
        A = [img] + [None] * L
        for l in range(L):
            A[l + 1] = layers[l].forward(A[l])

        scores = np.array(A[-1].data.view(-1))
        ind = np.argsort(-scores)
        top_class = ind[0]

        save_path = os.path.join(lrp_dir, cls_name, img_name.split('_')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_image(A[0].squeeze(), save_path + '/' + img_name + '.jpg')

        if imageClass[top_class] != cls_name:
            line = cls_name + '_' + img_name + ': Classification error! It should be ' + cls_name + ', but ' + imageClass[top_class]
            print(line)
            with open(save_path + '/' + img_name + '.txt', 'w') as f:
                f.write(line)

        T = torch.FloatTensor((1.0 * (np.arange(num_class) == top_class).reshape([1, num_class, 1, 1])))
        R = [None] * L + [(A[-1] * T).data]

        for l in range(1, L)[::-1]:
            A[l] = (A[l].data).requires_grad_(True)

            if isinstance(layers[l], torch.nn.MaxPool2d):
                layers[l] = torch.nn.AvgPool2d(2)

            if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):
                if l <= 16:
                    rho = lambda p: p + 0.25 * p.clamp(min=0)
                    incr = lambda z: z + 1e-9
                if 17 <= l <= 30:
                    rho = lambda p: p
                    incr = lambda z: z + 1e-9 + 0.25 * ((z**2).mean()**.5).data
                if l >= 31:
                    rho = lambda p: p
                    incr = lambda z: z + 1e-9

                z = incr(newlayer(layers[l], rho).forward(A[l]))         # step 1
                s = (R[l + 1] / z).data                                  # step 2
                (z * s).sum().backward()                                 # step 3
                c = A[l].grad
                R[l] = (A[l] * c).data                                   # step 4

            else:
                R[l] = R[l + 1]

        R1 = np.array(R[1][0]).sum(axis=0)
        R_dict.update({'f' + img_name: R1})

        obeservation_list = [
            # 31, 21, 11,
            1
        ]

        for i, l in enumerate(obeservation_list):
            heatmap(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)
            # plt.savefig(save_path + '/' + task + '.png')
        break
        # plt.close('all')
    # savemat(os.path.join(save_dir, 'R1.mat'), R_dict)
