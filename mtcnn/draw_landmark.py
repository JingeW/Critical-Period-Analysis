from PIL import Image
from src.detector import detect_faces
from src.utils import show_bboxes

img_path = '/home/sda1/Jinge/Attention_analysis/data/CASIA_WebFace_20000/test/0000102/321.jpg'
landmark_path = img_path.replace('test', 'test_landmark').replace('jpg', 'pkl')
image_class = img_path.split('/')[-2]
image_name = img_path.split('/')[-1]
print('[>] Now doing...', image_class, image_name)
image = Image.open(img_path).convert('RGB')
bounding_boxes, landmarks = detect_faces(image)
image_copy = show_bboxes(image, bounding_boxes, landmarks)

image_copy.save('/home/sda1/Jinge/Attention_analysis/result/mtcnn_sample_result.jpg')