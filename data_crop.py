import pickle as pkl
from PIL import Image

task = 'mouth'
img_path = f'/home/sda1/Jinge/Attention_analysis/result/{task}Only_bluring_sample_result.jpg'
image = Image.open(img_path).convert("RGB")

bbox_path = '/home/sda1/Jinge/Attention_analysis/data/CASIA_WebFace_20000/test_bbox/0000102/321.pkl'
with open(bbox_path, 'rb') as f:
    bbox = pkl.load(f)

bbox = bbox.squeeze()
im_crop = image.crop(
    (bbox[0], bbox[1], bbox[2], bbox[3],)
)
save_path = img_path.replace('_bluring', '_bluring_crop')
im_crop.save(save_path)