import os
import traceback
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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


def get_df(log_dir, adjusment=True):
    df = tflog2pandas(log_dir)
    df = df[(df.metric) == 'acc/valid']
    df = df['value'] 
    if adjusment:
        step = log_dir.split('_')[-1].split(']')[0]
        df = list_adjustment_new(df, step)
    return df

def get_df_mouth_model_fill(log_dir, df_mouth):
    df = tflog2pandas(log_dir)
    df = df[(df.metric) == 'acc/valid']
    df = df['value']
    df_full = np.concatenate([df_mouth[:80], df])
    return df_full


# Color scheme
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([1, 1, 1, 1, 1])
# base models
C_full_base = '#E31A1C'
C_eyes_base = '#3274a1'
C_mouth_base = '#e1812c'

# plt.figure()
# plt.plot(x, y, color=C_full_base)
# plt.plot(x, y*2 , color=C_eyes_base)
# plt.plot(x, y*3, color=C_mouth_base)

# In/out
C_inD = '#8D109D'
C_in = '#AF58BA'
C_inL = '#D78CE0'

C_outD = '#333333'
C_out = '#666666'
C_outL = '#999999'

# plt.figure()
# plt.plot(x, y*4, color=C_in)
# plt.plot(x, y*5, color=C_out)

# Recovery
C_Impared = 'k'
C_LR01 = '#009392'
C_LR05 = '#39B1B5'
C_LR001 = '#9CCB86'
C_AT = '#045275'

# plt.figure()
# plt.plot(x, y*6, color=C_Impared, linestyle='dotted')
# plt.plot(x, y*9, color=C_LR01)
# plt.plot(x, y*8, color=C_LR05)
# plt.plot(x, y*7, color=C_LR001)
# plt.plot(x, y*10, color=C_AT)

log_root = './logs/CASIA_WebFace_20000_0.15_final/'
save_dir = './result_new_acc_new/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Base models
full_base_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]'
full_base_df = get_df(full_base_dir, False)

eyes_base_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]'
eyes_base_df = get_df(eyes_base_dir, False)

mouth_base_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]'
mouth_base_df = get_df(mouth_base_dir, False)

plt.figure()
plt.plot(full_base_df, label='Full face', color=C_full_base)
plt.plot(eyes_base_df, label='Eyes foveated', color=C_eyes_base)
plt.plot(mouth_base_df, label='Mouth foveated', color=C_mouth_base)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of different models')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'baseModel_performance.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'baseModel_performance.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'baseModel_performance.svg', bbox_inches='tight', dpi=100)

# face on region model
# eyes
full_eyes15_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_15]'
full_eyes15_df = get_df(full_eyes15_dir)

full_eyes80_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_80]'
full_eyes80_df = get_df(full_eyes80_dir)

plt.figure()
plt.plot(eyes_base_df, label='Eyes foveated base model', color=C_eyes_base)
plt.plot(full_eyes15_df, label='Within Critical Period', color=C_in)
plt.plot(full_eyes80_df, label='Outside Critical Period', color=C_out)

plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of fullFace on eyes models')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullOnEyes_performance.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnEyes_performance.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnEyes_performance.svg', bbox_inches='tight', dpi=100)

# mouth
full_mouth15_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_15]'
full_mouth15_df = get_df(full_mouth15_dir)

full_mouth80_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
full_mouth80_df = get_df(full_mouth80_dir)

plt.figure()
plt.plot(mouth_base_df, label='Mouth foveated base model', color=C_mouth_base)
plt.plot(full_mouth15_df, label='Within Critical Period', color=C_in)
plt.plot(full_mouth80_df, label='Outside Critical Period', color=C_out)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of fullFace on mouth models')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullOnMouth_performance.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnMouth_performance.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnMouth_performance.svg', bbox_inches='tight', dpi=100)

# bluring on region model
# eyesBased
mouth_eyes15_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[eyes0.15_15]'
mouth_eyes15_df = get_df(mouth_eyes15_dir)

mouth_eyes80_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[eyes0.15_80]'
mouth_eyes80_df = get_df(mouth_eyes80_dir)

plt.figure()
plt.plot(eyes_base_df, label='Eyes foveated base model', color=C_eyes_base)
plt.plot(mouth_eyes15_df, label='Within Critical Period', color=C_in)
plt.plot(mouth_eyes80_df, label='Outside Critical Period', color=C_out)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of mouthFace on eyes models')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'mouthOnEyes_performance.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'mouthOnEyes_performance.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'mouthOnEyes_performance.svg', bbox_inches='tight', dpi=100)

# mouthBased
eyes_mouth15_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_15]'
eyes_mouth15_df = get_df(eyes_mouth15_dir)

eyes_mouth80_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_80]'
eyes_mouth80_df = get_df(eyes_mouth80_dir)

plt.figure()
plt.plot(mouth_base_df, label='Mouth foveated base model', color=C_mouth_base)
plt.plot(eyes_mouth15_df, label='Within Critical Period', color=C_in)
plt.plot(eyes_mouth80_df, label='Outside Critical Period', color=C_out)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of eyesFace on mouth models')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'eyesOnMouth_performance.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'eyesOnMouth_performance.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'eyesOnMouth_performance.svg', bbox_inches='tight', dpi=100)

# recovery by LR
log_root = './logs/CASIA_WebFace_20000_0.15_final_recover/'

eyes_base_lr_path = '/home/sda1/Jinge/Critical_Period_Analysis/logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]/lr_dict.pkl'
with open(eyes_base_lr_path, 'rb') as f:
    base_lr_list = pkl.load(f) 
base_lr_list = list(base_lr_list.values())
plt.figure()
plt.plot(base_lr_list, color=C_eyes_base)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.title('LR of eyes base model')
plt.savefig(save_dir + f'eyes_baseModel_LR.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'eyes_baseModel_LR.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'eyes_baseModel_LR.svg', bbox_inches='tight', dpi=100)

mouth_base_lr_path = '/home/sda1/Jinge/Critical_Period_Analysis/logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]/lr_dict.pkl'
with open(mouth_base_lr_path, 'rb') as f:
    base_lr_list = pkl.load(f) 
base_lr_list = list(base_lr_list.values())
plt.figure()
plt.plot(base_lr_list, color=C_mouth_base)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.title('LR of mouth base model')
plt.savefig(save_dir + f'mouth_baseModel_LR.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'mouth_baseModel_LR.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'mouth_baseModel_LR.svg', bbox_inches='tight', dpi=100)

# recovery LR
log_root = './logs/CASIA_WebFace_20000_0.15_final_recover/'

lr1_dir = '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
path1 = log_root + lr1_dir
# df1 = get_df(path1)
df1 = get_df_mouth_model_fill(path1, mouth_base_df)

lr2_dir = '[resnet50]_[0.005]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
path2 = log_root + lr2_dir
# df2 = get_df(path2)
df2 = get_df_mouth_model_fill(path2, mouth_base_df)

lr3_dir = '[resnet50]_[0.001]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
path3 = log_root + lr3_dir
# df3 = get_df(path3)
df3 = get_df_mouth_model_fill(path3, mouth_base_df)

path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
# df = get_df(path)
df = get_df_mouth_model_fill(path, mouth_base_df)

plt.figure()
plt.plot(df1, label='LR0.01_recovery', color= C_LR01)
plt.plot(df2, label='LR0.005_recovery', color=C_LR05)
plt.plot(df3, label='LR0.001_recovery', color=C_LR001)
plt.plot(full_base_df, label='Full face base model', color=C_full_base)
plt.plot(eyes_mouth15_df, label='Within Critical Period', color=C_in)
plt.plot(df, label='Impared', color='k', linestyle='dotted')
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.xlim(75, 155)
plt.ylim(0.68, 0.89)
plt.title('fullFace on mouth models recovery by LR')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullOnMouth_recovery_LR_focus.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnMouth_recovery_LR_focus.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnMouth_recovery_LR_focus.svg', bbox_inches='tight', dpi=100)

# recovery AT
path = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
df = get_df(path)

recovery_path = './logs/CASIA_WebFace_20000_0.15_final_recovery_v1/[resnet50]_[4]_[1]_[10]_[mouth0.15_80]'
recovery_df = tflog2pandas(recovery_path)
recovery_df = recovery_df[(recovery_df.metric) == 'acc/valid']
recovery_df = recovery_df['value'] 
recovery_df = list_adjustment_new(recovery_df, 80)

plt.figure()
plt.plot(recovery_df, label='AT_Recovery', color=C_AT)
plt.plot(full_base_df, label='Full face base model', color=C_full_base)
plt.plot(eyes_mouth15_df, label='Within Critical Period', color=C_in)
plt.plot(df, label='Impared', color='k', linestyle='dotted')
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.xlim(75, 155)
plt.ylim(0.68, 0.89)
plt.title('fullFace on mouth models recovery by AT')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullOnMouth_recovery_AT.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnMouth_recovery_AT.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnMouth_recovery_AT.svg', bbox_inches='tight', dpi=100)

# Performance for different model
log_root = './logs/CASIA_WebFace_20000_0.15_final_other/'

# Base models
full_base_dir = log_root + '[vgg16]_[0.001]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]'
full_base_df = get_df(full_base_dir, False)

eyes_base_dir = log_root + '[vgg16]_[0.001]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]'
eyes_base_df = get_df(eyes_base_dir, False)

mouth_base_dir = log_root + '[vgg16]_[0.001]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]'
mouth_base_df = get_df(mouth_base_dir, False)

plt.figure()
plt.plot(full_base_df, label='Full face', color=C_full_base)
plt.plot(eyes_base_df, label='Eyes foveated', color=C_eyes_base)
plt.plot(mouth_base_df, label='Mouth foveated', color=C_mouth_base)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of different models on vgg16')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'baseModel_performance_vgg16.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'baseModel_performance_vgg16.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'baseModel_performance_vgg16.svg', bbox_inches='tight', dpi=100)

# eyesBased
mouth_eyes15_dir = log_root + '[vgg16]_[0.001]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[eyes0.15_15]'
mouth_eyes15_df = get_df(mouth_eyes15_dir)

mouth_eyes80_dir = log_root + '[vgg16]_[0.001]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[eyes0.15_80]'
mouth_eyes80_df = get_df(mouth_eyes80_dir)

plt.figure()
plt.plot(eyes_base_df, label='Eyes foveated base model', color=C_eyes_base)
plt.plot(mouth_eyes15_df, label='Within Critical Period', color=C_in)
plt.plot(mouth_eyes80_df, label='Outside Critical Period', color=C_out)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend(loc='lower right')
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of mouthFace on eyes models on vgg16')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'mouthOnEyes_performance_vgg16.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'mouthOnEyes_performance_vgg16.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'mouthOnEyes_performance_vgg16.svg', bbox_inches='tight', dpi=100)

# mouthBased
eyes_mouth15_dir = log_root + '[vgg16]_[0.001]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_15]'
eyes_mouth15_df = get_df(eyes_mouth15_dir)

eyes_mouth80_dir = log_root + '[vgg16]_[0.001]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[mouth0.15_80]'
eyes_mouth80_df = get_df(eyes_mouth80_dir)

plt.figure()
plt.plot(mouth_base_df, label='Mouth foveated base model', color=C_mouth_base)
plt.plot(eyes_mouth15_df, label='Within Critical Period', color=C_in)
plt.plot(eyes_mouth80_df, label='Outside  Critical Period', color=C_out)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend(loc='lower right')
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of eyesFace on mouth models on vgg16')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'eyesOnMouth_performance_vgg16.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'eyesOnMouth_performance_vgg16.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'eyesOnMouth_performance_vgg16.svg', bbox_inches='tight', dpi=100)

# CP different epoch
# eyes
log_root = './logs/CASIA_WebFace_20000_0.15_final/'
eyes_base_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]'
eyes_base_df = get_df(eyes_base_dir, False)

full_eyes10_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_10]'
full_eyes10_df = get_df(full_eyes10_dir)

full_eyes15_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_15]'
full_eyes15_df = get_df(full_eyes15_dir)

full_eyes20_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_20]'
full_eyes20_df = get_df(full_eyes20_dir)

full_eyes70_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_70]'
full_eyes70_df = get_df(full_eyes70_dir)

full_eyes80_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_80]'
full_eyes80_df = get_df(full_eyes80_dir)

full_eyes90_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[eyes0.15_90]'
full_eyes90_df = get_df(full_eyes90_dir)

plt.figure()
plt.plot(eyes_base_df, label='Eyes foveated base model', color=C_eyes_base)
plt.plot(full_eyes10_df, label='Epoch 10(Within)', color=C_inD)
plt.plot(full_eyes15_df, label='Epoch 15(Within)', color=C_in)
plt.plot(full_eyes20_df, label='Epoch 20(Within)', color=C_inL)
plt.plot(full_eyes70_df, label='Epoch 70(Outside)', color=C_outD)
plt.plot(full_eyes80_df, label='Epoch 80(Outside)', color=C_out)
plt.plot(full_eyes90_df, label='Epoch 90(Outside)', color=C_outL)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of fullFace on eyes models with diff epoch')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullOnEyes_performance_diffEpoch.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnEyes_performance_diffEpoch.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnEyes_performance_diffEpoch.svg', bbox_inches='tight', dpi=100)

# mouth
mouth_base_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]'
mouth_base_df = get_df(mouth_base_dir, False)

full_mouth10_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_10]'
full_mouth10_df = get_df(full_mouth10_dir)

full_mouth15_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_15]'
full_mouth15_df = get_df(full_mouth15_dir)

full_mouth20_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_20]'
full_mouth20_df = get_df(full_mouth20_dir)

full_mouth70_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_70]'
full_mouth70_df = get_df(full_mouth70_dir)

full_mouth80_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_80]'
full_mouth80_df = get_df(full_mouth80_dir)

full_mouth90_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[mouth0.15_90]'
full_mouth90_df = get_df(full_mouth90_dir)

plt.figure()
plt.plot(mouth_base_df, label='Mouth foveated base model', color=C_mouth_base)
plt.plot(full_mouth10_df, label='Epoch 10(Within)', color=C_inD)
plt.plot(full_mouth15_df, label='Epoch 15(Within)', color=C_in)
plt.plot(full_mouth20_df, label='Epoch 20(Within)', color=C_inL)
plt.plot(full_mouth70_df, label='Epoch 70(Outside)', color=C_outD)
plt.plot(full_mouth80_df, label='Epoch 80(Outside)', color=C_out)
plt.plot(full_mouth90_df, label='Epoch 90(Outside)',color=C_outL)
plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of fullFace on mouth models with diff epoch')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullOnMouth_performance_diffEpoch.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnMouth_performance_diffEpoch.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullOnMouth_performance_diffEpoch.svg', bbox_inches='tight', dpi=100)


# Different initiation LR
log_root = './logs/CASIA_WebFace_20000_0.15_final/'

base_dir = log_root + '[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]'
df_base = get_df(base_dir, False)
LR02_dir = log_root + '[resnet50]_[0.02]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]'
df_LR02 = get_df(LR02_dir, False)
LR05_dir = log_root + '[resnet50]_[0.05]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]'
df_LR05 = get_df(LR05_dir, False)
LR005_dir = log_root + '[resnet50]_[0.005]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]'
df_LR005 = get_df(LR005_dir, False)
plt.figure()
plt.plot(df_base, label='Full face base model (Init LR = 0.01)', color=C_full_base)
plt.plot(df_LR02, label='Initial LR = 0.02')
plt.plot(df_LR05, label='Initial LR = 0.05')
plt.plot(df_LR005, label='Initial LR = 0.005')


plt.axvspan(0, 30, alpha=0.2, color=C_in)
plt.legend()
plt.grid(axis='both', linestyle='--')
plt.ylim(0, 1)
plt.title('Performance of full face base model with diff Init LR')
plt.xlabel('Epoch')
plt.ylabel('Acurracy')
plt.savefig(save_dir + f'fullface_performance_diffInitLR.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullface_performance_diffInitLR.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'fullface_performance_diffInitLR.svg', bbox_inches='tight', dpi=100)
