import os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

def get_df_baseModel(path, metric_name):
    with open(path, 'rb') as f:
        metric = pkl.load(f)
    df = pd.DataFrame(metric, columns=metric_name)
    return df


def draw_boxplot_baseModel(df, metric_names, pal, title1, title2):
    # sns.set(rc={'figure.figsize':(4,6)})
    # sns.set_theme(style='white')
    fig, axes = plt.subplots(1, 2)
    sns.boxplot(
        data=df[[metric_names[0], metric_names[1]]], width=0.5, showmeans=True, 
        meanprops={'marker':'*','markerfacecolor':'white', 'markeredgecolor':'black'},
        palette=pal,
        ax=axes[0]
    )
    axes[0].set_xticklabels(['eyes', 'mouth'], rotation=0)
    axes[0].set(ylabel='Average Intensity')
    axes[0].grid(axis='y', alpha=0.5)
    x1, x2 = 0, 1
    y = max(df[metric_names[0]].max(), df[metric_names[1]].max())
    y = y + 0.1*y
    h = 0.02*y
    axes[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    axes[0].text((x1+x2)*.5, y+h, '***', ha='center', va='bottom', color='k')
    axes[0].set_title(title1)

    sns.boxplot(
        data=df[[metric_names[2], metric_names[3]]], width=0.5, showmeans=True, 
        meanprops={'marker':'*','markerfacecolor':'white', 'markeredgecolor':'black'},
        palette=pal,
        ax=axes[1]
    )
    axes[1].set_xticklabels(['eyes', 'mouth'], rotation=0)
    axes[1].set(ylabel='Intensity Proportion')
    axes[1].grid(axis='y', alpha=0.5)
    x1, x2 = 0, 1
    y = max(df[metric_names[2]].max(), df[metric_names[3]].max())
    y = y + 0.1*y
    h = 0.02*y
    axes[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    axes[1].text((x1+x2)*.5, y+h, '***', ha='center', va='bottom', color='k')
    axes[1].set_title(title2)


def get_df(path15, path80):
    with open(path15, 'rb') as f:
        metric15 = pkl.load(f)
    with open(path80, 'rb') as f:
        metric80 = pkl.load(f)

    metric_avgEyes = np.array([metric15[:, 0], metric80[:, 0]]).transpose()
    metric_avgMouth = np.array([metric15[:, 1], metric80[:, 1]]).transpose()
    metric_propEyes = np.array([metric15[:, 2], metric80[:, 2]]).transpose()
    metric_propMouth = np.array([metric15[:, 3], metric80[:, 3]]).transpose()

    df0 = pd.DataFrame(metric_avgEyes, columns=['Within', 'Outside']).assign(grp='Eyes')
    df1 = pd.DataFrame(metric_avgMouth, columns=['Within', 'Outside']).assign(grp='Mouth')
    df2 = pd.DataFrame(metric_propEyes, columns=['Within', 'Outside']).assign(grp='Eyes')
    df3 = pd.DataFrame(metric_propMouth, columns=['Within', 'Outside']).assign(grp='Mouth')

    return df0, df1, df2, df3


def get_stats(df0, df1, df2, df3):
    avg15_eyes, avg80_eyes = df0['Within'], df0['Outside']
    avg15_mouth, avg80_mouth = df1['Within'], df1['Outside']
    prop15_eyes, prop80_eyes = df2['Within'], df2['Outside']
    prop15_mouth, prop80_mouth = df3['Within'], df3['Outside']

    _, p1 = stats.ttest_ind(avg15_eyes, avg15_mouth)
    _, p2 = stats.ttest_ind(avg80_eyes, avg80_mouth)
    _, p3 = stats.ttest_rel(avg15_eyes, avg80_eyes)
    _, p4 = stats.ttest_rel(avg15_mouth, avg80_mouth)

    _, p5 = stats.ttest_ind(prop15_eyes, prop15_mouth)
    _, p6 = stats.ttest_ind(prop80_eyes, prop80_mouth)
    _, p7 = stats.ttest_rel(prop15_eyes, prop80_eyes)
    _, p8 = stats.ttest_rel(prop15_mouth, prop80_mouth)

    sig_avg = []
    for p in [p1, p2, p3, p4]:
        if  0.01 < p <= 0.05:
            sig_avg.append('*')
        elif 0.001< p <= 0.01:
            sig_avg.append('**')
        elif p <=0.001:
            sig_avg.append('***')
        else:
            sig_avg.append(None)

    sig_prop = []
    for p in [p5, p6, p7, p8]:
        if  0.01 < p <= 0.05:
            sig_prop.append('*')
        elif 0.001< p <= 0.01:
            sig_prop.append('**')
        elif p <=0.001:
            sig_prop.append('***')
        else:
            sig_prop.append(None)

    return sig_avg, sig_prop


def draw_boxplot(df1, df2, y_name, pal, test_stats):
    # merge and melt dfs
    cdf = pd.concat([df1, df2])
    mdf = pd.melt(cdf, id_vars=['grp'])

    # Draw boxplot
    plt.figure()
    ax = sns.boxplot(
        x="grp", y="value", hue='variable', data=mdf, width=0.5, showmeans=True,
        meanprops={'marker':'*','markerfacecolor':'white', 'markeredgecolor':'black'},
        palette=pal
        )
    ax.legend(title='', loc='upper right')
    ax.set(xlabel=None, ylabel=y_name)
    ax.grid(axis='y', alpha=0.5)

    # Draw test stats
    m = 0.125
    x1, x2, x3, x4 = 0-m, 0+m, 1-m, 1+m
    y = mdf['value'].max()
    h = 0.02*y

    if test_stats[0]:
        y1 = y+0.3*y
        ax.plot([x1, x1, x3, x3], [y1, y1+h, y1+h, y1], lw=1.5, c='k')
        ax.text((x1+x3)*.5, y1+h, test_stats[0], ha='center', va='bottom', color='k')

    if test_stats[1]:
        y2 = y+0.2*y
        ax.plot([x2, x2, x4, x4], [y2, y2+h, y2+h, y2], lw=1.5, c='k')
        ax.text((x2+x4)*.5, y2+h, test_stats[1], ha='center', va='bottom', color='k')

    if test_stats[2]:
        y3 = y+0.1*y
        ax.plot([x1, x1, x2, x2], [y3, y3+h, y3+h, y3], lw=1.5, c='k')
        ax.text((x1+x2)*.5, y3+h, test_stats[2], ha='center', va='bottom', color='k')

    if test_stats[3]:
        y4 = y+0.1*y
        ax.plot([x3, x3, x4, x4], [y4, y4+h, y4+h, y4], lw=1.5, c='k')
        ax.text((x3+x4)*.5, y4+h, test_stats[3], ha='center', va='bottom', color='k')


root = './stats_final/gradcam/'
save_dir = './result_new_boxplot/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# base model
full_base_path = './stats_final/gradcam/full_baseModel_metrics.pkl'
eyes_base_path = './stats_final/gradcam/eyes_baseModel_metrics.pkl'
mouth_base_path = './stats_final/gradcam/mouth_baseModel_metrics.pkl'
metric_name = ['Avg_eyes', 'Avg_mouth', 'Prop_eyes', 'Prop_mouth']
my_pal = ['#3274a1', '#e1812c']

df_full = get_df_baseModel(full_base_path, metric_name)
df_eyes = get_df_baseModel(eyes_base_path, metric_name)
df_mouth = get_df_baseModel(mouth_base_path, metric_name)

# draw_boxplot_baseModel(df_full, 'Avg_eyes', 'Avg_mouth', my_pal, 'Average Intensity')
name_avg = 'full_baseModel_Avg'
# plt.title(name_avg)
# plt.savefig(save_dir + f'{name_avg}.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_avg}.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_avg}.svg', bbox_inches='tight', dpi=100)
# draw_boxplot_baseModel(df_full, 'Prop_eyes', 'Prop_mouth', my_pal, 'Intensity Proportion')
name_prop = 'full_baseModel_Prop'
# plt.title(name_avg)
# plt.savefig(save_dir + f'{name_prop}.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_prop}.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_prop}.svg', bbox_inches='tight', dpi=100)

draw_boxplot_baseModel(df_full, metric_name, my_pal, name_avg, name_prop)
plt.subplots_adjust(wspace=0.4, hspace=0.1)
plt.savefig(save_dir + 'full_baseModel.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + 'full_baseModel.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + 'full_baseModel.svg', bbox_inches='tight', dpi=100)


# draw_boxplot_baseModel(df_eyes,  my_pal, 'Average Intensity')
name_avg = 'eyes_baseModel_Avg'
# plt.title(name_avg)
# plt.savefig(save_dir + f'{name_avg}.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_avg}.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_avg}.svg', bbox_inches='tight', dpi=100)
# draw_boxplot_baseModel(df_eyes, 'Prop_eyes', 'Prop_mouth', my_pal, 'Intensity Proportion')
name_prop = 'eyes_baseModel_prop'
# plt.title(name_avg)
# plt.savefig(save_dir + f'{name_prop}.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_prop}.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_prop}.svg', bbox_inches='tight', dpi=100)

draw_boxplot_baseModel(df_eyes, metric_name, my_pal, name_avg, name_prop)
plt.subplots_adjust(wspace=0.4, hspace=0.1)
plt.savefig(save_dir + 'eyes_baseModel.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + 'eyes_baseModel.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + 'eyes_baseModel.svg', bbox_inches='tight', dpi=100)

# draw_boxplot_baseModel(df_mouth, 'Avg_eyes', 'Avg_mouth', my_pal, 'Average Intensity')
name_avg = 'mouth_baseModel_Avg'
# plt.title(name_avg)
# plt.savefig(save_dir + f'{name_avg}.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_avg}.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_avg}.svg', bbox_inches='tight', dpi=100)
# draw_boxplot_baseModel(df_mouth, 'Prop_eyes', 'Prop_mouth', my_pal, 'Intensity Proportion')
name_prop = 'mouth_baseModel_Prop'
# plt.title(name_avg)
# plt.savefig(save_dir + f'{name_prop}.png', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_prop}.eps', bbox_inches='tight', dpi=100)
# plt.savefig(save_dir + f'{name_prop}.svg', bbox_inches='tight', dpi=100)

draw_boxplot_baseModel(df_mouth, metric_name, my_pal, name_avg, name_prop)
plt.subplots_adjust(wspace=0.4, hspace=0.1)
plt.savefig(save_dir + 'mouth_baseModel.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + 'mouth_baseModel.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + 'mouth_baseModel.svg', bbox_inches='tight', dpi=100)


# full on eyes model
fullOnEyes15_path = './stats_final/gradcam/full_on_eyes15_metrics.pkl'
fullOnEyes80_path = './stats_final/gradcam/full_on_eyes80_metrics.pkl'
my_pal = ['#AF58BA', '#666666']

df0, df1, df2, df3 = get_df(fullOnEyes15_path, fullOnEyes80_path)
sig_avg, sig_prop = get_stats(df0, df1, df2, df3)

draw_boxplot(df0, df1, 'Average Intensity', my_pal, test_stats=sig_avg)
name_avg = 'fullFace_on_eyesModel_Avg'
plt.title(name_avg)
plt.savefig(save_dir + f'{name_avg}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.svg', bbox_inches='tight', dpi=100)

draw_boxplot(df2, df3, 'Intensity Proportion', my_pal, test_stats=sig_prop)
name_prop = 'fullFace_on_eyesModel_prop'
plt.title(name_prop)
plt.savefig(save_dir + f'{name_prop}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.svg', bbox_inches='tight', dpi=100)

# full on mouth model
fullOnMouth15_path = './stats_final/gradcam/full_on_mouth15_metrics.pkl'
fullOnMouth80_path = './stats_final/gradcam/full_on_mouth80_metrics.pkl'

df0, df1, df2, df3 = get_df(fullOnMouth15_path, fullOnMouth80_path)
sig_avg, sig_prop = get_stats(df0, df1, df2, df3)

draw_boxplot(df0, df1, 'Average Intensity', my_pal, test_stats=sig_avg)
name_avg = 'fullFace_on_mouthModel_Avg'
plt.title(name_avg)
plt.savefig(save_dir + f'{name_avg}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.svg', bbox_inches='tight', dpi=100)

draw_boxplot(df2, df3, 'Intensity Proportion', my_pal, test_stats=sig_prop)
name_prop = 'fullFace_on_mouthModel_prop'
plt.title(name_prop)
plt.savefig(save_dir + f'{name_prop}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.svg', bbox_inches='tight', dpi=100)


# mouth on eyes model
mouthOnEyes15_path = './stats_final/gradcam/mouth_on_eyes15_metrics.pkl'
mouthOnEyes80_path = './stats_final/gradcam/mouth_on_eyes80_metrics.pkl'

df0, df1, df2, df3 = get_df(mouthOnEyes15_path, mouthOnEyes80_path)
sig_avg, sig_prop = get_stats(df0, df1, df2, df3)

draw_boxplot(df0, df1, 'Average Intensity', my_pal, test_stats=sig_avg)
name_avg = 'mouthFace_on_eyesModel_Avg'
plt.title(name_avg)
plt.savefig(save_dir + f'{name_avg}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.svg', bbox_inches='tight', dpi=100)

draw_boxplot(df2, df3, 'Intensity Proportion', my_pal, test_stats=sig_prop)
name_prop = 'mouthFace_on_eyesModel_prop'
plt.title(name_prop)
plt.savefig(save_dir + f'{name_prop}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.svg', bbox_inches='tight', dpi=100)


# eyes on mouth model
eyesOnMouth15_path = './stats_final/gradcam/eyes_on_mouth15_metrics.pkl'
eyesOnMouth80_path = './stats_final/gradcam/eyes_on_mouth80_metrics.pkl'

df0, df1, df2, df3 = get_df(eyesOnMouth15_path, eyesOnMouth80_path)
sig_avg, sig_prop = get_stats(df0, df1, df2, df3)

draw_boxplot(df0, df1, 'Average Intensity', my_pal, test_stats=sig_avg)
name_avg = 'eyesFace_on_mouthModel_Avg'
plt.title(name_avg)
plt.savefig(save_dir + f'{name_avg}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.svg', bbox_inches='tight', dpi=100)

draw_boxplot(df2, df3, 'Intensity Proportion', my_pal, test_stats=sig_prop)
name_prop = 'eyesFace_on_mouthModel_prop'
plt.title(name_prop)
plt.savefig(save_dir + f'{name_prop}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.svg', bbox_inches='tight', dpi=100)



