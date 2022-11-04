import os
import traceback
from matplotlib.lines import lineStyles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def list_adjustment(array, labels):
    res = []
    for list_, label in zip(array, labels):
        if label == 'baseModel':
            list_ = list(list_[:150])
            print(label, len(list_))
        else:
            list_ = [np.nan] * int(label) + list(list_)
            print(label, len(list_))
        res.append(list_)
    return res


def list_adjustment_new(array, epoch):
    return [np.nan] * int(epoch) + list(array)


log_root = './logs/CASIA_WebFace_20000_0.15_final/'
save_dir = './result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
log_dir_list = sorted([log_dir for log_dir in os.listdir(log_root) if 'vgg' not in log_dir])
print(log_dir_list)

# cross bases
full_base_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]'
full_base_df = tflog2pandas(full_base_dir)
full_base_df = full_base_df[(full_base_df.metric) == 'acc/valid']
full_base_df = full_base_df['value']
eyes_base_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]'
eyes_base_df = tflog2pandas(eyes_base_dir)
eyes_base_df = eyes_base_df[(eyes_base_df.metric) == 'acc/valid']
eyes_base_df = eyes_base_df['value']
mouth_base_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]'
mouth_base_df = tflog2pandas(mouth_base_dir)
mouth_base_df = mouth_base_df[(mouth_base_df.metric) == 'acc/valid']
mouth_base_df = mouth_base_df['value']

plt.figure()
plt.plot(full_base_df, label='fullFace', color='C3')
plt.plot(eyes_base_df, label='eyesOnly', color='C0')
plt.plot(mouth_base_df, label='mouthOnly', color='C1')
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of different models')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'baseModel_performance.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'baseModel_performance.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'baseModel_performance.svg', bbox_inches='tight', dpi=100)


# face on region model
# eyes
full_eyes15_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_15]'
full_eyes15_df = tflog2pandas(full_eyes15_dir)
full_eyes15_df = full_eyes15_df[(full_eyes15_df.metric) == 'acc/valid']
full_eyes15_df = full_eyes15_df['value'] 
full_eyes15_df = list_adjustment_new(full_eyes15_df, 15)

full_eyes80_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_80]'
full_eyes80_df = tflog2pandas(full_eyes80_dir)
full_eyes80_df = full_eyes80_df[(full_eyes80_df.metric) == 'acc/valid']
full_eyes80_df = full_eyes80_df['value'] 
full_eyes80_df = list_adjustment_new(full_eyes80_df, 80)

plt.figure()
plt.plot(eyes_base_df, label='baseModel', color='k', linestyle='--')
plt.plot(full_eyes15_df, label='In Critical Period', color='C3')
plt.plot(full_eyes80_df, label='Out of Critical Period', color='C0')
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of fullFace on eyes models')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullOnEyes_performance.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'fullOnEyes_performance.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'fullOnEyes_performance.svg', bbox_inches='tight', dpi=100)

# mouth
full_mouth15_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_15]'
full_mouth15_df = tflog2pandas(full_mouth15_dir)
full_mouth15_df = full_mouth15_df[(full_mouth15_df.metric) == 'acc/valid']
full_mouth15_df = full_mouth15_df['value'] 
full_mouth15_df = list_adjustment_new(full_mouth15_df, 15)

full_mouth80_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
full_mouth80_df = tflog2pandas(full_mouth80_dir)
full_mouth80_df = full_mouth80_df[(full_mouth80_df.metric) == 'acc/valid']
full_mouth80_df = full_mouth80_df['value'] 
full_mouth80_df = list_adjustment_new(full_mouth80_df, 80)

plt.figure()
plt.plot(mouth_base_df, label='baseModel', color='k', linestyle='--')
plt.plot(full_mouth15_df, label='In Critical Period', color='C3')
plt.plot(full_mouth80_df, label='Out of Critical Period', color='C0')
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of fullFace on mouth models')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullOnMouth_performance.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'fullOnMouth_performance.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'fullOnMouth_performance.svg', bbox_inches='tight', dpi=100)

# bluring on region model
# eyesBased
mouth_eyes15_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[eyes0.15_15]'
mouth_eyes15_df = tflog2pandas(mouth_eyes15_dir)
mouth_eyes15_df = mouth_eyes15_df[(mouth_eyes15_df.metric) == 'acc/valid']
mouth_eyes15_df = mouth_eyes15_df['value'] 
mouth_eyes15_df = list_adjustment_new(mouth_eyes15_df, 15)

mouth_eyes80_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[eyes0.15_80]'
mouth_eyes80_df = tflog2pandas(mouth_eyes80_dir)
mouth_eyes80_df = mouth_eyes80_df[(mouth_eyes80_df.metric) == 'acc/valid']
mouth_eyes80_df = mouth_eyes80_df['value'] 
mouth_eyes80_df = list_adjustment_new(mouth_eyes80_df, 80)

plt.figure()
plt.plot(eyes_base_df, label='baseModel', color='k', linestyle='--')
plt.plot(mouth_eyes15_df, label='In Critical Period', color='C3')
plt.plot(mouth_eyes80_df, label='Out of Critical Period', color='C0')
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of mouthFace on eyes models')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'mouthOnEyes_performance.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'mouthOnEyes_performance.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'mouthOnEyes_performance.svg', bbox_inches='tight', dpi=100)

# mouthBased
eyes_mouth15_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_15]'
eyes_mouth15_df = tflog2pandas(eyes_mouth15_dir)
eyes_mouth15_df = eyes_mouth15_df[(eyes_mouth15_df.metric) == 'acc/valid']
eyes_mouth15_df = eyes_mouth15_df['value'] 
eyes_mouth15_df = list_adjustment_new(eyes_mouth15_df, 15)

eyes_mouth80_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_80]'
eyes_mouth80_df = tflog2pandas(eyes_mouth80_dir)
eyes_mouth80_df = eyes_mouth80_df[(eyes_mouth80_df.metric) == 'acc/valid']
eyes_mouth80_df = eyes_mouth80_df['value'] 
eyes_mouth80_df = list_adjustment_new(eyes_mouth80_df, 80)

plt.figure()
plt.plot(mouth_base_df, label='baseModel', color='k', linestyle='--')
plt.plot(eyes_mouth15_df, label='In Critical Period', color='C3')
plt.plot(eyes_mouth80_df, label='Out of Critical Period', color='C0')
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of eyesFace on mouth models')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'eyesOnMouth_performance.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'eyesOnMouth_performance.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'eyesOnMouth_performance.svg', bbox_inches='tight', dpi=100)

# recovery
import pickle as pkl
log_root = './logs/CASIA_WebFace_20000_0.15_final_recover/'
save_dir = './result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
log_dir_list = sorted([log_dir for log_dir in os.listdir(log_root) if 'vgg' not in log_dir])
print(log_dir_list)

base_lr_path = '/home/sda1/Jinge/Attention_analysis/logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]/lr_dict.pkl'
with open(base_lr_path, 'rb') as f:
    base_lr_list = pkl.load(f) 
base_lr_list = list(base_lr_list.values())
plt.figure()
plt.plot(base_lr_list)
plt.title('LR of mouth base model')
plt.savefig(save_dir + f'mouth_baseModel_LR.png', bbox_inches='tight', dpi=100)

lr1_dir = '[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_80]'
path1 = log_root + lr1_dir
df1 = tflog2pandas(path1)
df1 = df1[(df1.metric) == 'acc/valid']
df1 = df1['value'] 

lr2_dir = '[resnet50]_[0.005]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_80]'
path2 = log_root + lr2_dir
df2 = tflog2pandas(path2)
df2 = df2[(df2.metric) == 'acc/valid']
df2 = df2['value'] 

lr3_dir = '[resnet50]_[0.001]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_80]'
path3 = log_root + lr3_dir
df3 = tflog2pandas(path3)
df3 = df3[(df3.metric) == 'acc/valid']
df3 = df3['value'] 

plt.figure()
plt.plot(df1, label='0.01', color= 'C3')
plt.plot(df2, label='0.005', color='C0')
plt.plot(df3, label='0.001', color='C1')
plt.legend()
plt.grid(axis='both', linestyle='--')
# plt.ylim(0, 1)
plt.title('Performance of eyesFace on mouth models recovery')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'OnMouth_recovery_performance.png', bbox_inches='tight', dpi=100)


# recovery0
import pickle as pkl
log_root = './logs/CASIA_WebFace_20000_0.15_final_recover/'
save_dir = './result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
log_dir_list = sorted([log_dir for log_dir in os.listdir(log_root) if 'vgg' not in log_dir])
print(log_dir_list)

# base_lr_path = '/home/sda1/Jinge/Attention_analysis/logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]/lr_dict.pkl'
# with open(base_lr_path, 'rb') as f:
#     base_lr_list = pkl.load(f) 
# base_lr_list = list(base_lr_list.values())
# plt.figure()
# plt.plot(base_lr_list)
# plt.title('LR of mouth base model')
# plt.savefig(save_dir + f'mouth_baseModel_LR.png', bbox_inches='tight', dpi=100)

lr1_dir = '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
path1 = log_root + lr1_dir
df1 = tflog2pandas(path1)
df1 = df1[(df1.metric) == 'acc/valid']
df1 = df1['value']
df1 = list_adjustment_new(df1, 80)

lr2_dir = '[resnet50]_[0.005]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
path2 = log_root + lr2_dir
df2 = tflog2pandas(path2)
df2 = df2[(df2.metric) == 'acc/valid']
df2 = df2['value']
df2 = list_adjustment_new(df2, 80)

lr3_dir = '[resnet50]_[0.001]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
path3 = log_root + lr3_dir
df3 = tflog2pandas(path3)
df3 = df3[(df3.metric) == 'acc/valid']
df3 = df3['value']
df3 = list_adjustment_new(df3, 80)

path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
df = tflog2pandas(path)
df = df[(df.metric) == 'acc/valid']
df = df['value'] 
df = list_adjustment_new(df, 80)

recovery_path = './logs/CASIA_WebFace_20000_0.15_final_recovery_v1/[resnet50]_[4]_[1]_[10]_[mouth0.15_80]'
recovery_df = tflog2pandas(recovery_path)
recovery_df = recovery_df[(recovery_df.metric) == 'acc/valid']
recovery_df = recovery_df['value'] 
recovery_df = list_adjustment_new(recovery_df, 80)

plt.figure()
plt.plot(df1, label='LR0.01_recovery', color= 'C3')
plt.plot(df2, label='LR0.005_recovery', color='C0')
plt.plot(df3, label='LR0.001_recovery', color='C1')
plt.plot(recovery_df, label='AT_Recovery', color='m', linestyle='dashdot')
plt.plot(df, label='Deficit', color='k', linestyle='dotted')
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.xlim(75, 155)
plt.ylim(0.7, 0.85)
plt.title('fullFace on mouth models recovery by LR')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullOnMouth_recovery.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'fullOnMouth_recovery_LR.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'fullOnMouth_recovery_LR.svg', bbox_inches='tight', dpi=100)

# recovery1
import pickle as pkl
path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
df = tflog2pandas(path)
df = df[(df.metric) == 'acc/valid']
df = df['value'] 
df = list_adjustment_new(df, 80)

recovery_path = './logs/CASIA_WebFace_20000_0.15_final_recovery_v1/[resnet50]_[4]_[1]_[10]_[mouth0.15_80]'
recovery_df = tflog2pandas(recovery_path)
recovery_df = recovery_df[(recovery_df.metric) == 'acc/valid']
recovery_df = recovery_df['value'] 
recovery_df = list_adjustment_new(recovery_df, 80)

plt.figure()
plt.plot(recovery_df, label='AT_Recovery', color='C3')
plt.plot(df, label='Deficit', color='k', linestyle='--')
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.xlim(75, 155)
# plt.ylim(0, 1)
plt.title('fullFace on mouth models recovery by AT')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullOnMouth_recovery_AT.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'fullOnMouth_recovery_AT.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'fullOnMouth_recovery_AT.svg', bbox_inches='tight', dpi=100)



# mouth_idx = [4, 1, 2, 3]
# eyes_idx = [0, 5, 6, 7]
# df_list = []
# for log_dir in log_dir_list:
#     path = os.path.join(log_root, log_dir)
#     df=tflog2pandas(path)
#     df = df[(df.metric) == 'acc/valid']
#     df_list.append(df['value'])
# df_array = np.array(df_list)

# labels = ['baseModel', '15', '80']

# y_mouth = df_array[mouth_idx]
# y_mouth_new = list_adjustment(y_mouth, labels)
# y_eyes = df_array[eyes_idx]
# y_eyes_new = list_adjustment(y_eyes, labels)

# # Mouth based
# plt.figure()
# for y, l in zip(y_mouth_new, labels):
#     plt.plot(y, label=l)
# plt.legend()
# plt.grid(axis='both', linestyle='--')
# plt.ylim(0.4, 0.9)
# plt.title('Mouth only base')
# plt.xlabel('Epoch')
# plt.ylabel('Acurracy')
# plt.savefig(save_dir + f'Mouth_based.png', bbox_inches='tight', dpi=100)
# # plt.savefig(save_dir + f'Mouth_based.eps', bbox_inches='tight', dpi=100)
# # plt.savefig(save_dir + f'Mouth_based.svg', bbox_inches='tight', dpi=100)

# # Eyes based
# plt.figure()
# for y, l in zip(y_eyes_new, labels):
#     plt.plot(y, label=l)
# plt.legend()
# plt.grid(axis='both', linestyle='--')
# plt.ylim(0.4, 0.9)
# plt.title('Eyes only base')
# plt.xlabel('Epoch')
# plt.ylabel('Acurracy')
# plt.savefig(save_dir + f'Eyes_based.png', bbox_inches='tight', dpi=100)
# # plt.savefig(save_dir + f'Eyes_based.eps', bbox_inches='tight', dpi=100)
# # plt.savefig(save_dir + f'Eyes_based.svg', bbox_inches='tight', dpi=100)


