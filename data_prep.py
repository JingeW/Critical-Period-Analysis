import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt

# data_root = './data'
# source_dir = os.path.join(data_root, 'CelebA500_cropped')
# dest_dir = os.path.join(data_root, 'CelebA_face_new_cropped')
# if not os.path.exists(dest_dir):
#     os.makedirs(dest_dir)

# file_list = []
# for root, _, filenames in os.walk(source_dir):
#     for filename in filenames:
#         file_list.append(os.path.join(root, filename))

# for source in file_list:
#     shutil.copy(source, dest_dir)


data_root = './data/CASIA_WebFace_20000/train'
file_list= []
for root, _, filenames in os.walk(data_root):
    for filename in filenames:
        file_list.append(os.path.join(root, filename))

# plt.figure(1, figsize=(8, 2))
# plt.axis('off')
# n = 0
# for i in range(16):
#   n += 1
#   random_img = random.choice(file_list)
#   imgs = Image.open(random_img)
#   plt.subplot(2, 8, n)
#   plt.axis('off')
#   plt.imshow(imgs)
# plt.subplots_adjust(wspace=0, hspace=0)

# plt.savefig('/home/sda1/Jinge/Attention_analysis/result/dataset_sample.jpg')

dest = './data/exampler'
if not os.path.exists(dest):
    os.makedirs(dest)
for i in range(16):
  random_img = random.choice(file_list)
  shutil.copy(random_img, dest)