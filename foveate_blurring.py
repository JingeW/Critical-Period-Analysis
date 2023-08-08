import cv2
import numpy as np
import os
import pickle as pkl
from PIL import Image


def genGaussiankernel(width, sigma):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d

def pyramid(im, sigma=1, prNum=6):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]
    
    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma)
    
    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width/2), int(height/2)))
        pyramids.append(G)
    
    # upsample
    for i in range(1, prNum):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i-1:
                im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im

    return pyramids

def foveat_img(im, fixs, p, k, alpha, sigma, prNum):
    """
    im: input image
    fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    
    This function outputs the foveated image with given input image and fixations.
    """
    # sigma = 0.248
    # prNum = 5
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape
    print('height, width', height, width)

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)
    # print('theta', theta, len(theta), len(theta[0]))
    # np.savetxt('./theta.csv', theta, delimiter=',')
    
    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(theta))
    # print('Ts', Ts)

    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma

    omega[omega>1] = 1

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))
    # print('Bs', Bs)

    # M
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))

    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i-1][ind]

        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]
    # print('Ms', Ms)

    print('num of full-res pixel', np.sum(Ms[0] == 1))
    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])

    im_fov = im_fov.astype(np.uint8)
    return im_fov

def get_img(img_dir):
    img_dir_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
    print(len(img_dir_list))
    img_path_list = []
    for img_dir in img_dir_list:
        img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        img_path_list.extend(img_path)
    return img_path_list

def get_img_new(img_dir):
    img_path_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)], key=str.casefold)
    return img_path_list


if __name__ == "__main__":
    # task = 'forehead'

    # img_path = './data/CASIA_WebFace_20000/test/0004937/321.jpg'
    # landmark_path = img_path.replace('test', 'test_landmark').replace('jpg', 'pkl')
    # im = cv2.imread(img_path)
    # with open(landmark_path, 'rb') as f:
    #     landmark_list = pkl.load(f)
    # landmark_list = landmark_list[0]
    # if  task == 'eyes':
    #     fixes = [
    #         (landmark_list[0], landmark_list[5]),
    #         (landmark_list[1], landmark_list[6]),
    #         ((landmark_list[0] + landmark_list[1]) / 2, (landmark_list[5] + landmark_list[6]) / 2)
    #     ]
    # elif task == 'mouth':
    #     fixes = [
    #         (landmark_list[3] , landmark_list[8]),
    #         (landmark_list[4], landmark_list[9]),
    #         ((landmark_list[3] + landmark_list[4]) / 2, (landmark_list[8] + landmark_list[9]) / 2)
    #     ]
    # elif task == 'forehead':
    #     # dist = abs(landmark_list[0] - landmark_list[1]) / 4 * 3
    #     dist = 45
    #     print(dist)
    #     fixes = [
    #         (landmark_list[0], landmark_list[5] - dist),
    #         (landmark_list[1], landmark_list[6] - dist),
    #         ((landmark_list[0] + landmark_list[1]) / 2, ((landmark_list[5] + landmark_list[6]) / 2) - dist)
    #     ]
    # else:
    #     fixes =[
    #         (landmark_list[0], landmark_list[5]),
    #         (landmark_list[1], landmark_list[6]),
    #         ((landmark_list[0] + landmark_list[1]) / 2, (landmark_list[5] + landmark_list[6]) / 2),
    #         (landmark_list[3] , landmark_list[8]),
    #         (landmark_list[4], landmark_list[9]),
    #         ((landmark_list[3] + landmark_list[4]) / 2, (landmark_list[8] + landmark_list[9]) / 2)
    #     ]

    # coef = [0.1, 7, 2, 0.15] # [0.15, 0.5, 2, 2], [0.1, 7, 2, 0.15], [0.1, 7, 2, 0.2]
    # sigma = coef[0]     # higher = more blur
    # p = coef[1]         # higher = less blur
    # k = coef[2]         # higher = less blur
    # alpha = coef[3]     # higher = less blur
    # prNum = 8

    # img = foveat_img(im, fixes, p, k, alpha, sigma, prNum)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_rgb = Image.fromarray(img_rgb)
    # img_rgb.save(f'/home/sda1/Jinge/Attention_analysis/result/{task}Only_bluring_sample_result.jpg')

    data_dir = './data/CASIA_WebFace_20000'
    dataset_name = 'test'  # 'train', 'test'
    image_dir = os.path.join(data_dir, dataset_name)
    task = 'forehead' # 'mouth', 'eyes', 'both', 'forehead'

    if task == 'mouth':
        save_dir = os.path.join(data_dir, dataset_name + '_mouth')
    elif task == 'eyes':
        save_dir = os.path.join(data_dir, dataset_name + '_eyes')
    elif task == 'forehead':
        save_dir = os.path.join(data_dir, dataset_name + '_forehead')  
    else:
        save_dir = os.path.join(data_dir, dataset_name + '_both')

    image_path_list = get_img(image_dir)
    log = open(data_dir + '/failed_file.txt', 'w')
    for image_path in image_path_list[:]:
        image_class = image_path.split('/')[-2]
        image_name = image_path.split('/')[-1]
        print('[>] Now doing...', image_class, image_name)

        landmark_path = image_path.replace('jpg', 'pkl').replace(dataset_name, dataset_name + '_landmark')
        with open(landmark_path, 'rb') as f:
            landmark_list = pkl.load(f)

        if len(landmark_list) == 0:
            log.write(image_path  +'\n')
            continue
        else:
            landmark_list = landmark_list[0]
            if  task == 'eyes':
                fixes = [
                    (landmark_list[0], landmark_list[5]),
                    (landmark_list[1], landmark_list[6]),
                    ((landmark_list[0] + landmark_list[1]) / 2, (landmark_list[5] + landmark_list[6]) / 2)
                    # ((landmark_list[0] + landmark_list[1]) / 2, (landmark_list[5] + landmark_list[6]) / 2 - abs(landmark_list[0] - landmark_list[1])/4)
                ]
            elif task =='mouth':
                fixes = [
                    (landmark_list[3] , landmark_list[8]),
                    (landmark_list[4], landmark_list[9]),
                    ((landmark_list[3] + landmark_list[4]) / 2, (landmark_list[8] + landmark_list[9]) / 2)
                    # ((landmark_list[3] + landmark_list[4]) / 2, (landmark_list[8] + landmark_list[9]) / 2 + abs(landmark_list[3] - landmark_list[4])/4)
                ]
            elif task == 'forehead':
                dist = 50  # 45
                fixes = [
                    (landmark_list[0], landmark_list[5] - dist),
                    (landmark_list[1], landmark_list[6] - dist),
                    ((landmark_list[0] + landmark_list[1]) / 2, ((landmark_list[5] + landmark_list[6]) / 2) - dist)
                ]
            else:
                fixes =[
                    (landmark_list[0], landmark_list[5]),
                    (landmark_list[1], landmark_list[6]),
                    ((landmark_list[0] + landmark_list[1]) / 2, (landmark_list[5] + landmark_list[6]) / 2),
                    (landmark_list[3] , landmark_list[8]),
                    (landmark_list[4], landmark_list[9]),
                    ((landmark_list[3] + landmark_list[4]) / 2, (landmark_list[8] + landmark_list[9]) / 2)
                ]

            coef = [0.1, 7, 2, 0.12] # [0.15, 0.5, 2, 2], [0.1, 7, 2, 0.15], [0.1, 7, 2, 0.2]
            sigma = coef[0]     # higher = more blur
            p = coef[1]         # higher = less blur
            k = coef[2]         # higher = less blur
            alpha = coef[3]     # higher = less blur
            prNum = 8

            im = cv2.imread(image_path)
            img = foveat_img(im, fixes, p, k, alpha, sigma, prNum)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            a = Image.fromarray(img_rgb)

            save_subDir = os.path.join(save_dir + str(alpha) + '_' + str(dist), image_class)
            if not os.path.exists(save_subDir):
                os.makedirs(save_subDir)
            cv2.imwrite(os.path.join(save_subDir, image_name), img)
    log.close()