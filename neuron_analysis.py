import os 
import numpy as np
import pickle as pkl

data_dir = './result_neuronSelection/'
# model_name_list = ['fullFace_baseModel', 'eyesFoveated_baseModel', 'mouthFoveated_baseModel']
# model_name_list = ['fullOnEyes_epoch15', 'fullOnEyes_epoch80', 'fullOnMouth_epoch15', 'fullOnMouth_epoch80']
# model_name_list = ['foreheadFoveated_baseModel', 'foreheadFoveated_50_baseModel']
# model_name_list = [ 'fullOnForehead_epoch15', 'fullOnForehead_epoch80', 'fullOnForehead50_epoch15', 'fullOnForehead50_epoch80']
model_name_list = [
'fullFace_baseModel', 'eyesFoveated_baseModel', 'foreheadFoveated_baseModel', 
'fullOnEyes_epoch15', 'fullOnEyes_epoch80',
'fullOnForehead_epoch15', 'fullOnForehead_epoch80'
]

for model_name in model_name_list:
    file_dir = data_dir + model_name

    file_list = sorted([os.path.join(file_dir, f) for f in os.listdir(file_dir) if 'ind' in f])

    sig_dict = {}
    nonSig_dict = {}
    for path in file_list:
        name = path.split('/')[-1]
        layer = name.split('_')[0]

        with open(path, 'rb') as f:
            ind_list = pkl.load(f)

        if 'non' in name:
            nonSig_dict.update({layer: len(ind_list)})
        else:
            sig_dict.update({layer: len(ind_list)})

    print(model_name)
    # print('        sig nonSig')
    for idx, tup in enumerate(zip(sig_dict.values(), nonSig_dict.values())):
        print(f'Layer{idx}: {tup[0]} {sum(tup)}', '{:.0%}'.format(tup[0]/sum(tup)))
        # print(f'Layer{idx}:', '{:.0%}'.format(tup[0]/sum(tup)), '{:.0%}'.format(tup[1]/sum(tup)))

