import imp
import matplotlib.pyplot as plt
import numpy as np
import os
import innvestigate
import innvestigate.utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import preprocess_input
import keras
import torch
from scipy.io import savemat
from PIL import Image

base_dir = os.path.dirname(__file__)
utils = imp.load_source("utils", os.path.join(base_dir, "utils_lrp.py"))


def alexnet_model(img_shape=(224, 224, 3), n_classes=50):

    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(64, (11, 11), strides=(4,4), input_shape=img_shape, padding='valid'))
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # Layer 2
    alexnet.add(ZeroPadding2D((2, 2)))
    alexnet.add(Conv2D(192, (5, 5), strides=(1,1),  padding='valid'))
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(384, (3, 3), strides=(1,1), padding='valid'))
    alexnet.add(Activation('relu'))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(256, (3, 3), strides=(1,1), padding='valid'))
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(256, (3, 3), strides=(1,1), padding='valid'))
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dropout(0.5))
    alexnet.add(Dense(4096))
    alexnet.add(Activation('relu'))

    # Layer 7
    alexnet.add(Dropout(0.5))
    alexnet.add(Dense(4096))
    alexnet.add(Activation('relu'))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(Activation('softmax'))

    return alexnet


if __name__ == "__main__":
    data_dir = "./data/CASIA_WebFace_5000"
    data_name = 'test_crop'
    epoch = '40'
    img_dir = os.path.join(data_dir, data_name)
    log_dir = f'./logs/CASIA_WebFace_5000_0.2/[alex]_[0.001]_[0.5]_[train_crop_mouth0.2_{epoch}]'
    task = f'train_crop_mouth0.2_{epoch}'
    lrp_dir = './lrp_result_new/' + task
    if not os.path.exists(lrp_dir):
        os.makedirs(lrp_dir)

    pytorch_pretrain_ckpt = os.path.join(log_dir, '200.pth')
    pytorch_para_dict = torch.load(pytorch_pretrain_ckpt)

    keras_model = keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), activation='relu', padding="valid"),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="valid"),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="valid"),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="valid"),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(50, activation='softmax')
    ])

    keras_model = alexnet_model()
    keras_model.summary()
    keras_weights = keras_model.get_weights()
    layer_list = [
        'features.0', 'features.3', 
        'features.6', 'features.8', 'features.10', 
        'classifier.1', 'classifier.4', 'classifier.6'
    ]
    for i, layer in enumerate(layer_list):
        ind = 2 * i
        print(f'\n[+] weights index: {ind}, layer name: {layer}')
        weight_key = layer + '.weight'
        bias_key = layer + '.bias'
        print(f'[-] pytorch weight {pytorch_para_dict[weight_key].shape} and keras weight {keras_weights[ind].shape}')
        keras_weights[ind] = np.transpose(pytorch_para_dict[weight_key].cpu().numpy())
        print(f'[-] pytorch bias {pytorch_para_dict[bias_key].shape} and keras bias {keras_weights[ind+1].shape}')
        keras_weights[ind+1] = np.transpose(pytorch_para_dict[bias_key].cpu().numpy())
    keras_model.set_weights(keras_weights)
    keras_model = innvestigate.utils.model_wo_softmax(keras_model)

    lrp_tag = 'lrp.sequential_preset_a_flat'
    analyzer = innvestigate.create_analyzer(
        lrp_tag, keras_model,
        neuron_selection_mode="index"
    )

    image_path = '/home/sda1/Jinge/Attention_analysis/data/CASIA_WebFace_5000/test_crop/0005095/071.jpg'
    image = utils.load_image(image_path, 224)
    x = preprocess_input(image[None])

    result = analyzer.analyze(x, neuron_selection=0)
    # Aggregate along color channels and normalize to [-1, 1]
    a = result.sum(axis=np.argmax(np.asarray(result.shape) == 3))
    a /= np.max(np.abs(a))

    # R_dict.update({'f' + img_name: a[0]})

    plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
    plt.title(lrp_tag)