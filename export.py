import os
import pickle as pkl
import scipy.io as io

res_dir = './result_neuronSelection/'
export_dir = './export/'


name_list = ['fullFace_baseModel', 'eyesFoveated_baseModel', 'mouthFoveated_baseModel']

for name in name_list:
    save_dir = export_dir + name + '_fullFM/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pkl_dir = res_dir + name
    pkl_list = [os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir)]
    for pkl_path in pkl_list:
        with open(pkl_path, 'rb') as f:
            content = pkl.load(f)
        if 'matrix' in pkl_path:
            key = 'FM'
        else:
            key = 'idx'
        save_path = save_dir + pkl_path.split('/')[-1].replace('.pkl', '.mat')
        io.savemat(save_path, mdict={key: content})