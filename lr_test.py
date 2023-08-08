# import matplotlib.pyplot as plt
# import numpy as np
import pickle as pkl


# gamma_list = [0.9]
# print('ExponentialLR:')
# for gamma in gamma_list:
#     a = 1e-2
#     lr = [a]
#     for i in range(200):
#         a *= gamma
#         lr.append(a)

#     # plt.plot(lr)
#     print('gamma:', gamma)
#     m = 10
#     n = 60
#     print(f'LR of epoch {m}:', lr[m-1])
#     print(f'LR of epoch {n}:', lr[n-1])

# path = '/home/sda1/Jinge/Attention_analysis/logs/CASIA_WebFace_20000_0.15/[resnet50]_[0.01]_[0.5]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]/lr_dict.pkl'
# with open(path, 'rb') as f:
#     a = pkl.load(f)
# a = list(a.values())
# # plt.plot(a[:150])
# print('\nReduceLROnPlateau:')
# print('LR of epoch 15:', a[14])
# print('LR of epoch 80:', a[79])


# import os
# path1 = './data/CASIA_WebFace_20000/test_crop'
# path2 = './data/CASIA_WebFace_20000/train_crop'
# res1 = []
# for r, d, f in os.walk(path1):
#     for file in f:
#         if file.endswith('.jpg'):
#             res1.append(os.path.join(r, file))
# res2 = []
# for r, d, f in os.walk(path2):
#     for file in f:
#         if file.endswith('.jpg'):
#             res2.append(os.path.join(r, file))

# len(res1) + len(res2)

import numpy as np
import matplotlib.pyplot as plt

def learning_rate_decay(init_lr, gamma, epoch):
    return init_lr * (gamma ** epoch)

init_lr = 1e-2
gamma = 0.9
total_epochs = 150

learning_rates = [learning_rate_decay(init_lr, gamma, epoch) for epoch in range(total_epochs)]

plt.plot(learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('ExponentialLR Decay Curve')
# plt.grid(True)
plt.show()

def learning_rate_decay(init_lr, gamma, epoch, decay_interval):
    return init_lr * (gamma ** (epoch // decay_interval))

init_lr = 1e-2
gamma = 0.5
total_epochs = 150
decay_interval = 30

learning_rates = [learning_rate_decay(init_lr, gamma, epoch, decay_interval) for epoch in range(total_epochs)]

plt.plot(learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('StepLR Decay Curve')
# plt.grid(True)
plt.show()

path = '/home/sda1/Jinge/Critical_Period_Analysis/logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]/lr_dict.pkl'
with open(path, 'rb') as f:
    lr_dict = pkl.load(f)

plt.plot(list(lr_dict.values()))
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Adaptive Decay Curve')