import os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


root = './stats_final/gradcam/'
x1, x2 = 0, 1
col = 'k'
save_dir = './result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# path = './stats_final/gradcam/full_baseModel_metrics.pkl'
# path_list = sorted([os.path.join(root, f) for f in os.listdir(root) if 'metric' in f])
# for path in path_list:
#     name = path.split('/')[-1].split('.')[0]
#     print(name)
#     with open(path, 'rb') as f:
#         metric = pkl.load(f)
#     metric_name = ['Avg_eyes', 'Avg_mouth', 'Prop_eyes', 'Prop_mouth']
#     df = pd.DataFrame(metric, columns=metric_name)
#     mean_Avg_eyes, std_Avg_eyes = df['Avg_eyes'].mean(), df['Avg_eyes'].std()
#     mean_Avg_mouth, std_Avg_mouth = df['Avg_mouth'].mean(), df['Avg_mouth'].std()
#     mean_Prop_eyes, std_Prop_eyes = df['Prop_eyes'].mean(), df['Prop_eyes'].std()
#     mean_Prop_mouth, std_Prop_mouth = df['Prop_mouth'].mean(), df['Prop_mouth'].std()

#     print('Eyes mean: %.2f std: %.2f' % (mean_Avg_eyes, std_Avg_eyes), 'Mouth mean: %.2f std: %.2f' % (mean_Avg_mouth, std_Avg_mouth))
#     print('Eyes prop: %.2f std: %.2f' % (mean_Prop_eyes, std_Prop_eyes),'Mouth prop: %.2f std: %.2f' % (mean_Prop_mouth, std_Prop_mouth))
#     t1, p1 = stats.ttest_rel(df['Avg_eyes'], df['Avg_mouth'])
#     print(f'Avg T-stats:{t1}, P-value:{p1}')
#     t2, p2 = stats.ttest_rel(df['Prop_eyes'], df['Prop_mouth'])
#     print(f'Prop T-stats:{t2}, P-value:{p2}')

#     fig, axes = plt.subplots(1, 2)
#     sns.boxplot(
#         data=df[['Avg_eyes', 'Avg_mouth']], width=0.5,
#         showmeans=True, 
#         meanprops={'marker':'*','markerfacecolor':'white', 'markeredgecolor':'black'},
#         ax=axes[0]
#     )
    
#     axes[0].set_xticklabels(['eyes', 'mouth'], rotation=0)
#     axes[0].set(ylabel='Average Intensity')
#     axes[0].grid(alpha=0.5)

#     y = max(df[metric_name[0]].max(), df[metric_name[1]].max())
#     y = y + 0.1*y
#     h = 0.02*y
#     axes[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     axes[0].text((x1+x2)*.5, y+h, '***', ha='center', va='bottom', color=col)
#     # plt.savefig(save_dir + f'{name}_avg.png', bbox_inches='tight', dpi=100)

#     sns.boxplot(
#         data=df[['Prop_eyes', 'Prop_mouth']], showmeans=True, width=0.5,
#         meanprops={'marker':'*','markerfacecolor':'white', 'markeredgecolor':'black'},
#         ax=axes[1]
#     )
#     axes[1].set_xticklabels(['eyes', 'mouth'], rotation=0)
#     axes[1].set(ylabel='Intensity Proportion')
#     axes[1].yaxis.set_label_position("right")
#     axes[1].yaxis.tick_right()
#     axes[1].grid(alpha=0.5)

#     y = max(df[metric_name[2]].max(), df[metric_name[3]].max())
#     y = y + 0.1*y
#     h = 0.02*y
#     axes[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     axes[1].text((x1+x2)*.5, y+h, '***', ha='center', va='bottom', color=col)
#     # plt.savefig(save_dir + f'{name}_prop.png', bbox_inches='tight', dpi=100)
#     plt.subplots_adjust(wspace=0.01, hspace=0.1)
#     plt.suptitle(name)
#     plt.savefig(save_dir + f'{name}.png', bbox_inches='tight', dpi=100)

with open('./result/eyesRegion_pixel_list.pkl', 'rb') as f:
    eyesRegion_pixel_list = pkl.load(f)

with open('./result/mouthRegion_pixel_list.pkl', 'rb') as f:
    mouthRegion_pixel_list = pkl.load(f)

eyesRegion_pixel_list = np.array(eyesRegion_pixel_list)
mouthRegion_pixel_list = np.array(mouthRegion_pixel_list)
print('Eyes mean: %.2f' % eyesRegion_pixel_list.mean(), 'std: %.2f' % eyesRegion_pixel_list.std())
print('Mouth mean: %.2f' % mouthRegion_pixel_list.mean(),'std:%.2f' % mouthRegion_pixel_list.std())

stats.ttest_ind(eyesRegion_pixel_list, mouthRegion_pixel_list)

def get_df(path):
    with open(path, 'rb') as f:
        metric = pkl.load(f)
    metric_name = ['Avg_eyes', 'Avg_mouth', 'Prop_eyes', 'Prop_mouth']
    df = pd.DataFrame(metric, columns=metric_name)
    return df

'''
Use ind_t test comparing eyes vs mouth in same model (DoF: 7536)
Use paired_t test comparing stats crossing models (DoF: 3768)
'''
# base model
full_path = './stats_final/gradcam/full_baseModel_metrics.pkl'
eyes_path = './stats_final/gradcam/eyes_baseModel_metrics.pkl'
mouth_path = './stats_final/gradcam/mouth_baseModel_metrics.pkl'
df_full = get_df(full_path)
df_eyes = get_df(eyes_path)
df_mouth = get_df(mouth_path)

# Base Line =====================================================
# average Grad-CAM intensity of full base model
df_full['Avg_eyes'].mean(), df_full['Avg_eyes'].std()
df_full['Avg_mouth'].mean(), df_full['Avg_mouth'].std()
# eyes vs mouth of average Grad-CAM intensity in full base model
stats.ttest_ind(df_full['Avg_eyes'], df_full['Avg_mouth'])
# proportion of Grad-CAM intensity of full base model
df_full['Prop_eyes'].mean(), df_full['Prop_eyes'].std()
df_full['Prop_mouth'].mean(), df_full['Prop_mouth'].std()
# eyes vs mouth of proportion of Grad-CAM intensity in full base model
stats.ttest_ind(df_full['Prop_eyes'], df_full['Prop_mouth'])

# average Grad-CAM intensity of eyes base model
df_eyes['Avg_eyes'].mean(), df_eyes['Avg_eyes'].std()
df_eyes['Avg_mouth'].mean(), df_eyes['Avg_mouth'].std()
# eyes vs mouth of average Grad-CAM intensity in eyes base model
stats.ttest_ind(df_eyes['Avg_eyes'], df_eyes['Avg_mouth'])
# proportion of Grad-CAM intensity of eyes base model
df_eyes['Prop_eyes'].mean(), df_eyes['Prop_eyes'].std()
df_eyes['Prop_mouth'].mean(), df_eyes['Prop_mouth'].std()
# eyes vs mouth of proportion of Grad-CAM intensity in eyes base model
stats.ttest_ind(df_eyes['Prop_eyes'], df_eyes['Prop_mouth'])

# average Grad-CAM intensity of mouth base model
df_mouth['Avg_eyes'].mean(), df_mouth['Avg_eyes'].std()
df_mouth['Avg_mouth'].mean(), df_mouth['Avg_mouth'].std()
# eyes vs mouth of average Grad-CAM intensity in mouth base model
stats.ttest_ind(df_mouth['Avg_eyes'], df_mouth['Avg_mouth'])
# proportion of Grad-CAM intensity of mouth base model
df_mouth['Prop_eyes'].mean(), df_mouth['Prop_eyes'].std()
df_mouth['Prop_mouth'].mean(), df_mouth['Prop_mouth'].std()
# eyes vs mouth of proportion of Grad-CAM intensity in mouth base model
stats.ttest_ind(df_mouth['Prop_eyes'], df_mouth['Prop_mouth'])

# eyes base vs full base on eyes info
stats.ttest_rel(df_eyes['Avg_eyes'], df_full['Avg_eyes'])
# mouth base vs full base on mouth info
stats.ttest_rel(df_mouth['Avg_mouth'], df_full['Avg_mouth'])
# ===============================================================

# Recovery with full-face images ================================
# +++ Full on eyes model
fullOnEyes15_path = './stats_final/gradcam/full_on_eyes15_metrics.pkl'
fullOnEyes80_path = './stats_final/gradcam/full_on_eyes80_metrics.pkl'
df_fullOnEyes15 = get_df(fullOnEyes15_path)
df_fullOnEyes80 = get_df(fullOnEyes80_path)

# average Grad-CAM intensity of mouth within and outside CP
df_fullOnEyes15['Avg_mouth'].mean(), df_fullOnEyes15['Avg_mouth'].std()
df_fullOnEyes80['Avg_mouth'].mean(), df_fullOnEyes80['Avg_mouth'].std()
# in vs out of mouth average Grad-CAM intensity
stats.ttest_rel(df_fullOnEyes15['Avg_mouth'], df_fullOnEyes80['Avg_mouth'])
# proportion of Grad-CAM intensity of mouth within and outside CP
df_fullOnEyes15['Prop_mouth'].mean(), df_fullOnEyes15['Prop_mouth'].std()
df_fullOnEyes80['Prop_mouth'].mean(), df_fullOnEyes80['Prop_mouth'].std()
# in vs out of mouth proportion of Grad-CAM intensity
stats.ttest_rel(df_fullOnEyes15['Prop_mouth'], df_fullOnEyes80['Prop_mouth'])

# eyes vs mouth within CP
stats.ttest_ind(df_fullOnEyes15['Avg_eyes'], df_fullOnEyes15['Avg_mouth'])
stats.ttest_ind(df_fullOnEyes15['Prop_eyes'], df_fullOnEyes15['Prop_mouth'])
# eyes vs mouth outside CP
stats.ttest_ind(df_fullOnEyes80['Avg_eyes'], df_fullOnEyes80['Avg_mouth'])
stats.ttest_ind(df_fullOnEyes80['Prop_eyes'], df_fullOnEyes80['Prop_mouth'])

# df_fullOnEyes15['Avg_eyes'].mean(), df_fullOnEyes15['Avg_eyes'].std()
# df_fullOnEyes80['Avg_eyes'].mean(), df_fullOnEyes80['Avg_eyes'].std()
# stats.ttest_rel(df_fullOnEyes15['Avg_eyes'], df_fullOnEyes80['Avg_eyes'])
# df_fullOnEyes15['Prop_eyes'].mean(), df_fullOnEyes15['Prop_eyes'].std()
# df_fullOnEyes80['Prop_eyes'].mean(), df_fullOnEyes80['Prop_eyes'].std()
# stats.ttest_rel(df_fullOnEyes15['Prop_eyes'], df_fullOnEyes80['Prop_eyes'])

# +++ Full on mouth model
fullOnMouth15_path = './stats_final/gradcam/full_on_mouth15_metrics.pkl'
fullOnMouth80_path = './stats_final/gradcam/full_on_mouth80_metrics.pkl'
df_fullOnMouth15 = get_df(fullOnMouth15_path)
df_fullOnMouth80 = get_df(fullOnMouth80_path)

# average Grad-CAM intensity of eyes within and outside CP
df_fullOnMouth15['Avg_eyes'].mean(), df_fullOnMouth15['Avg_eyes'].std()
df_fullOnMouth80['Avg_eyes'].mean(), df_fullOnMouth80['Avg_eyes'].std()
# in vs out of eyes average Grad-CAM intensity
stats.ttest_rel(df_fullOnMouth15['Avg_eyes'], df_fullOnMouth80['Avg_eyes'])
# proportion of Grad-CAM intensity of eyes within and outside CP
df_fullOnMouth15['Prop_eyes'].mean(), df_fullOnMouth15['Prop_eyes'].std()
df_fullOnMouth80['Prop_eyes'].mean(), df_fullOnMouth80['Prop_eyes'].std()
# in vs out of eyes proportion of Grad-CAM intensity
stats.ttest_rel(df_fullOnMouth15['Prop_eyes'], df_fullOnMouth80['Prop_eyes'])

# eyes vs mouth within CP
stats.ttest_ind(df_fullOnMouth15['Avg_eyes'], df_fullOnMouth15['Avg_mouth'])
stats.ttest_ind(df_fullOnMouth15['Prop_eyes'], df_fullOnMouth15['Prop_mouth'])
# eyes vs mouth outside CP
stats.ttest_ind(df_fullOnMouth80['Avg_eyes'], df_fullOnMouth80['Avg_mouth'])
stats.ttest_ind(df_fullOnMouth80['Prop_eyes'], df_fullOnMouth80['Prop_mouth'])

# df_fullOnMouth15['Avg_mouth'].mean(), df_fullOnMouth15['Avg_mouth'].std()
# df_fullOnMouth80['Avg_mouth'].mean(), df_fullOnMouth80['Avg_mouth'].std()
# stats.ttest_rel(df_fullOnMouth15['Avg_mouth'], df_fullOnMouth80['Avg_mouth'])
# df_fullOnMouth15['Prop_mouth'].mean(), df_fullOnMouth15['Prop_mouth'].std()
# df_fullOnMouth80['Prop_mouth'].mean(), df_fullOnMouth80['Prop_mouth'].std()
# stats.ttest_rel(df_fullOnMouth15['Prop_mouth'], df_fullOnMouth80['Prop_mouth'])

# +++ within / outside vs base model
# eyes base vs full base (85.67% vs. 87.32%)
stats.ttest_rel(df_fullOnEyes15['Avg_eyes'], df_full['Avg_eyes'])
stats.ttest_rel(df_fullOnEyes15['Avg_mouth'], df_full['Avg_mouth'])
stats.ttest_rel(df_fullOnEyes15['Prop_eyes'], df_full['Prop_eyes'])
stats.ttest_rel(df_fullOnEyes15['Prop_mouth'], df_full['Prop_mouth'])

# mouth base vs full base (84.66% vs. 87.32%)
stats.ttest_rel(df_fullOnMouth15['Avg_eyes'], df_full['Avg_eyes'])
stats.ttest_rel(df_fullOnMouth15['Avg_mouth'], df_full['Avg_mouth'])
stats.ttest_rel(df_fullOnMouth15['Prop_eyes'], df_full['Prop_eyes'])
stats.ttest_rel(df_fullOnMouth15['Prop_mouth'], df_full['Prop_mouth'])

# stats.ttest_rel(df_fullOnEyes80['Avg_eyes'], df_eyes['Avg_eyes'])
# stats.ttest_rel(df_fullOnEyes80['Avg_mouth'], df_eyes['Avg_mouth'])
# stats.ttest_rel(df_fullOnEyes80['Prop_eyes'], df_eyes['Prop_eyes'])
# stats.ttest_rel(df_fullOnEyes80['Prop_mouth'], df_eyes['Prop_mouth'])

# stats.ttest_rel(df_fullOnMouth80['Avg_eyes'], df_mouth['Avg_eyes'])
# stats.ttest_rel(df_fullOnMouth80['Avg_mouth'], df_mouth['Avg_mouth'])
# stats.ttest_rel(df_fullOnMouth80['Prop_eyes'], df_mouth['Prop_eyes'])
# stats.ttest_rel(df_fullOnMouth80['Prop_mouth'], df_mouth['Prop_mouth'])
# ================================================================

# Recovery with complementary information ========================
# +++ Mouth on eyes model
mouthOnEyes15_path = './stats_final/gradcam/mouth_on_eyes15_metrics.pkl'
mouthOnEyes80_path = './stats_final/gradcam/mouth_on_eyes80_metrics.pkl'
df_mouthOnEyes15 = get_df(mouthOnEyes15_path)
df_mouthOnEyes80 = get_df(mouthOnEyes80_path)

df_mouthOnEyes80['Avg_eyes'].mean(), df_mouthOnEyes80['Avg_eyes'].std()
df_mouthOnEyes80['Avg_mouth'].mean(), df_mouthOnEyes80['Avg_mouth'].std()
stats.ttest_ind(df_mouthOnEyes80['Avg_eyes'], df_mouthOnEyes80['Avg_mouth'])

df_mouthOnEyes80['Prop_eyes'].mean(), df_mouthOnEyes80['Prop_eyes'].std()
df_mouthOnEyes80['Prop_mouth'].mean(), df_mouthOnEyes80['Prop_mouth'].std()
stats.ttest_ind(df_mouthOnEyes80['Prop_eyes'], df_mouthOnEyes80['Prop_mouth'])

df_mouthOnEyes15['Avg_eyes'].mean(), df_mouthOnEyes15['Avg_eyes'].std()
df_mouthOnEyes15['Avg_mouth'].mean(), df_mouthOnEyes15['Avg_mouth'].std()
stats.ttest_ind(df_mouthOnEyes15['Avg_eyes'], df_mouthOnEyes15['Avg_mouth'])

df_mouthOnEyes15['Prop_eyes'].mean(), df_mouthOnEyes15['Prop_eyes'].std()
df_mouthOnEyes15['Prop_mouth'].mean(), df_mouthOnEyes15['Prop_mouth'].std()
stats.ttest_ind(df_mouthOnEyes15['Prop_eyes'], df_mouthOnEyes15['Prop_mouth'])

# stats.ttest_rel(df_mouthOnEyes80['Avg_eyes'], df_eyes['Avg_eyes'])
# stats.ttest_rel(df_mouthOnEyes80['Avg_mouth'], df_eyes['Avg_mouth'])
# stats.ttest_rel(df_mouthOnEyes80['Prop_eyes'], df_eyes['Prop_eyes'])
# stats.ttest_rel(df_mouthOnEyes80['Prop_mouth'], df_eyes['Prop_mouth'])

# stats.ttest_rel(df_mouthOnEyes80['Avg_eyes'], df_mouth['Avg_eyes'])
# stats.ttest_rel(df_mouthOnEyes80['Avg_mouth'], df_mouth['Avg_mouth'])
# stats.ttest_rel(df_mouthOnEyes80['Prop_eyes'], df_mouth['Prop_eyes'])
# stats.ttest_rel(df_mouthOnEyes80['Prop_mouth'], df_mouth['Prop_mouth'])

# +++ Eyes on mouth model
eyesOnMouth15_path = './stats_final/gradcam/eyes_on_mouth15_metrics.pkl'
eyesOnMouth80_path = './stats_final/gradcam/eyes_on_mouth80_metrics.pkl'
df_eyesOnMouth15 = get_df(eyesOnMouth15_path)
df_eyesOnMouth80 = get_df(eyesOnMouth80_path)

df_eyesOnMouth80['Avg_eyes'].mean(), df_eyesOnMouth80['Avg_eyes'].std()
df_eyesOnMouth80['Avg_mouth'].mean(), df_eyesOnMouth80['Avg_mouth'].std()
stats.ttest_ind(df_eyesOnMouth80['Avg_eyes'], df_eyesOnMouth80['Avg_mouth'])

df_eyesOnMouth80['Prop_eyes'].mean(), df_eyesOnMouth80['Prop_eyes'].std()
df_eyesOnMouth80['Prop_mouth'].mean(), df_eyesOnMouth80['Prop_mouth'].std()
stats.ttest_ind(df_eyesOnMouth80['Prop_eyes'], df_eyesOnMouth80['Prop_mouth'])

df_eyesOnMouth15['Avg_eyes'].mean(), df_eyesOnMouth15['Avg_eyes'].std()
df_eyesOnMouth15['Avg_mouth'].mean(), df_eyesOnMouth15['Avg_mouth'].std()
stats.ttest_ind(df_eyesOnMouth15['Avg_eyes'], df_eyesOnMouth15['Avg_mouth'])

df_eyesOnMouth15['Prop_eyes'].mean(), df_eyesOnMouth15['Prop_eyes'].std()
df_eyesOnMouth15['Prop_mouth'].mean(), df_eyesOnMouth15['Prop_mouth'].std()
stats.ttest_ind(df_eyesOnMouth15['Prop_eyes'], df_eyesOnMouth15['Prop_mouth'])

# stats.ttest_rel(df_eyesOnMouth80['Avg_eyes'], df_mouth['Avg_eyes'])
# stats.ttest_rel(df_eyesOnMouth80['Avg_mouth'], df_mouth['Avg_mouth'])
# stats.ttest_rel(df_eyesOnMouth80['Prop_eyes'], df_mouth['Prop_eyes'])
# stats.ttest_rel(df_eyesOnMouth80['Prop_mouth'], df_mouth['Prop_mouth'])

# stats.ttest_rel(df_eyesOnMouth15['Avg_eyes'], df_eyes['Avg_eyes'])
# stats.ttest_rel(df_eyesOnMouth15['Avg_mouth'], df_eyes['Avg_mouth'])
# stats.ttest_rel(df_eyesOnMouth15['Prop_eyes'], df_eyes['Prop_eyes'])
# stats.ttest_rel(df_eyesOnMouth15['Prop_mouth'], df_eyes['Prop_mouth'])

LR01_path = './stats_final/gradcam_recoveryLR/recoveryLR_0.01_metrics.pkl'
LR005_path = './stats_final/gradcam_recoveryLR/recoveryLR_0.005_metrics.pkl'
LR001_path = './stats_final/gradcam_recoveryLR/recoveryLR_0.001_metrics.pkl'
df_LR01 = get_df(LR01_path)
df_LR005 = get_df(LR005_path)
df_LR001 = get_df(LR001_path)
df_ANOVA_avg = pd.DataFrame({
    'Outside_CP': df_fullOnMouth80['Avg_eyes'],
    'LR0.001': df_LR001['Avg_eyes'],
    'LR0.005': df_LR005['Avg_eyes'],
    'LR0.01': df_LR01['Avg_eyes'],
    'Full_face': df_full['Avg_eyes']
})
df_ANOVA_avg.to_csv('./export/ANOVA_AVG.csv')
df_ANOVA_prop = pd.DataFrame({
    'Outside_CP': df_fullOnMouth80['Prop_eyes'],
    'LR0.001': df_LR001['Prop_eyes'],
    'LR0.005': df_LR005['Prop_eyes'],
    'LR0.01': df_LR01['Prop_eyes'],
    'Full_face': df_full['Prop_eyes']
})
df_ANOVA_prop.to_csv('./export/ANOVA_PROP.csv')

AT_path = './stats_final/gradcam_recovery_AT/recovery_4_1_10_metrics.pkl'
df_AT = get_df(AT_path)
df_fullOnMouth80['Avg_eyes'].mean(), df_fullOnMouth80['Avg_eyes'].std()
df_AT['Avg_eyes'].mean(), df_AT['Avg_eyes'].std()
stats.ttest_rel(df_fullOnMouth80['Avg_eyes'], df_AT['Avg_eyes'])

df_fullOnMouth80['Prop_eyes'].mean(), df_fullOnMouth80['Prop_eyes'].std()
df_AT['Prop_eyes'].mean(), df_AT['Prop_eyes'].std()
stats.ttest_rel(df_fullOnMouth80['Prop_eyes'], df_AT['Prop_eyes'])

stats.ttest_rel(df_LR001['Avg_eyes'], df_AT['Avg_eyes'])
stats.ttest_rel(df_LR001['Prop_eyes'], df_AT['Prop_eyes'])