import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


gamma_list = [0.9]
print('ExponentialLR:')
for gamma in gamma_list:
    a = 1e-2
    lr = [a]
    for i in range(200):
        a *= gamma
        lr.append(a)

    # plt.plot(lr)
    print('gamma:', gamma)
    m = 10
    n = 60
    print(f'LR of epoch {m}:', lr[m-1])
    print(f'LR of epoch {n}:', lr[n-1])

path = '/home/sda1/Jinge/Attention_analysis/logs/CASIA_WebFace_20000_0.15/[resnet50]_[0.01]_[0.5]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]/lr_dict.pkl'
with open(path, 'rb') as f:
    a = pkl.load(f)
a = list(a.values())
# plt.plot(a[:150])
print('\nReduceLROnPlateau:')
print('LR of epoch 15:', a[14])
print('LR of epoch 80:', a[79])