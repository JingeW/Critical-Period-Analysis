# import pickle as pkl
# from PIL import Image

# task = 'mouth'
# img_path = f'/home/sda1/Jinge/Attention_analysis/result/{task}Only_bluring_sample_result.jpg'
# image = Image.open(img_path).convert("RGB")

# bbox_path = '/home/sda1/Jinge/Attention_analysis/data/CASIA_WebFace_20000/test_bbox/0000102/321.pkl'
# with open(bbox_path, 'rb') as f:
#     bbox = pkl.load(f)

# bbox = bbox.squeeze()
# im_crop = image.crop(
#     (bbox[0], bbox[1], bbox[2], bbox[3],)
# )
# save_path = img_path.replace('_bluring', '_bluring_crop')
# im_crop.save(save_path)

import os
import numpy as np
import PIL.Image as Image
import pickle as pkl


def get_img(img_dir):
    img_dir_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
    img_path_list = []
    for img_dir in img_dir_list:
        img_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        img_path_list.extend(img_path)
    return img_path_list

def get_img_new(img_dir):
    img_path_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)], key=str.casefold)
    return img_path_list


data_dir = './data/CASIA_WebFace_20000'
dataset_name = 'test_forehead0.12_50'
image_dir = os.path.join(data_dir, dataset_name)

image_path_list = get_img(image_dir)
log = open(data_dir + '/'+ dataset_name + '_failed_bbox.txt', 'w')
for image_path in image_path_list:
    image_class = image_path.split('/')[-2]
    image_name = image_path.split('/')[-1]
    print('[>] Now doing...', image_class, image_name)

    image = Image.open(image_path).convert("RGB")
    bbox_path = image_path.replace('jpg', 'pkl').replace(dataset_name, dataset_name.split('_')[0] + '_bbox')
    with open(bbox_path, 'rb') as f:
        bbox = pkl.load(f)
    if len(bbox) != 1:
        log.write(image_path  +'\n')
        os.remove(image_path)
        continue
    else:
        bbox = bbox.squeeze()
        im_crop = image.crop(
            (bbox[0], bbox[1], bbox[2], bbox[3],)
        )

        save_dir = '/'.join(image_path.split('/')[:-1]).replace(dataset_name, dataset_name + '_crop')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = image_path.replace(dataset_name, dataset_name + '_crop')
        im_crop.save(save_path)
log.close()



        