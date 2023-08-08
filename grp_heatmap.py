import os
import cv2
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


def get_img(root):
    img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])[:10]
    # img_dir_list = sorted([os.path.join(root, f) for f in os.listdir(root)])
    img_path_list = []
    for img_dir in img_dir_list:
        img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))[:5]]
        # img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        img_path_list.extend(img_path)
    return img_path_list

# root = './stats_final/gradcam'
# root = './stats_final/gradcam_recovery_AT'
root = './stats_final/score_cam'
save_dir  = './result_cam_final/avg_heatmap'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
path_list = sorted([os.path.join(root, f) for f in os.listdir(root) if 'heatmap' in f])
hight, width = 180, 150
# face_image = cv2.imread('./data/generic_face_crop.jpg')
# face_image = cv2.resize(face_image, (width, hight))
face_image_dir = './data/CASIA_WebFace_20000/test_crop'
# face_image_list = []
# for root, dirs, files in os.walk(face_image_dir):
#     for name in files:
#         face_image_list.append(os.path.join(root, name))
face_image_list = get_img(face_image_dir)

face_list = []
for face_image in face_image_list:
    img = cv2.imread(face_image)
    img = cv2.resize(img, (width, hight))
    face_list.append(img)
face_list = np.array(face_list)
face_avg = face_list.mean(0)
# cv2.imwrite('./avgface.jpg', face_avg)
# face_avg = cv2.cvtColor(face_avg, cv2.COLOR_BGR2RGB)
# plt.imshow(face_avg)


for path in path_list[:]:
    name = path.split('/')[-1].split('.')[0].split('_h')[0]
    print(name)
    with open(path, 'rb') as f:
        heatmap = pkl.load(f)
        heatmap_resize = np.array([cv2.resize(hp, (width, hight)) for hp in heatmap])
        heatmap_avg = heatmap_resize.mean(0)
        heatmap_avg /= heatmap_avg.max()
        heatmap_avg = np.uint8(heatmap_avg*255)
        heatmap_avg = cv2.applyColorMap(heatmap_avg, cv2.COLORMAP_JET)
        superimposed_img = heatmap_avg * 0.4 + face_avg
        cv2.imwrite(save_dir + f'/{name}_avg_heatmap.jpg', superimposed_img)
        # heatmap_avg = cv2.cvtColor(heatmap_avg, cv2.COLOR_BGR2RGB)
        # plt.title(name)
        # plt.imshow(heatmap_avg)
        # plt.savefig(save_dir + f'/{name}_avg_heatmap.jpg')
        