import os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


def get_df(path_base, path01, path005, path001, path_80):
    with open(path_base, 'rb') as f:
        metric_base = pkl.load(f)
    with open(path01, 'rb') as f:
        metric01 = pkl.load(f)
    with open(path005, 'rb') as f:
        metric005 = pkl.load(f)
    with open(path001, 'rb') as f:
        metric001 = pkl.load(f)
    with open(path_80, 'rb') as f:
        metric_80 = pkl.load(f)

    metric_avgEyes = np.array(
        [metric_base[:, 0], metric01[:, 0], metric005[:, 0], metric001[:, 0], metric_80[:, 0]]
        ).transpose()
    metric_avgMouth = np.array(
        [metric_base[:, 1], metric01[:, 1], metric005[:, 1], metric001[:, 1], metric_80[:, 1]]
        ).transpose()
    metric_propEyes = np.array(
        [metric_base[:, 2], metric01[:, 2], metric005[:, 2], metric001[:, 2], metric_80[:, 2]]
        ).transpose()
    metric_propMouth = np.array(
        [metric_base[:, 3], metric01[:, 3], metric005[:, 3], metric001[:, 3], metric_80[:, 3]]
        ).transpose()

    df0 = pd.DataFrame(metric_avgEyes, columns=['Full_base', 'LR=0.01', 'LR=0.005', 'LR=0.001', 'Impaired']).assign(grp='Eyes')
    df1 = pd.DataFrame(metric_avgMouth, columns=['Full_base', 'LR=0.01', 'LR=0.005', 'LR=0.001', 'Impaired']).assign(grp='Mouth')
    df2 = pd.DataFrame(metric_propEyes, columns=['Full_base', 'LR=0.01', 'LR=0.005', 'LR=0.001', 'Impaired']).assign(grp='Eyes')
    df3 = pd.DataFrame(metric_propMouth, columns=['Full_base', 'LR=0.01', 'LR=0.005', 'LR=0.001', 'Impaired']).assign(grp='Mouth')
    return df0, df1, df2, df3


def get_df_AT(path_base, pathAT, path_80):
    with open(path_base, 'rb') as f:
        metric_base = pkl.load(f)
    with open(pathAT, 'rb') as f:
        metricAT = pkl.load(f)
    with open(path_80, 'rb') as f:
        metric_80 = pkl.load(f)

    metric_avgEyes = np.array(
        [metric_base[:, 0], metricAT[:, 0], metric_80[:, 0]]
        ).transpose()
    metric_avgMouth = np.array(
        [metric_base[:, 1], metricAT[:, 1], metric_80[:, 1]]
        ).transpose()
    metric_propEyes = np.array(
        [metric_base[:, 2], metricAT[:, 2], metric_80[:, 2]]
        ).transpose()
    metric_propMouth = np.array(
        [metric_base[:, 3], metricAT[:, 3], metric_80[:, 3]]
        ).transpose()

    df0 = pd.DataFrame(metric_avgEyes, columns=['Full_base', 'Attention Transfer', 'Impaired']).assign(grp='Eyes')
    df1 = pd.DataFrame(metric_avgMouth, columns=['Full_base', 'Attention Transfer', 'Impaired']).assign(grp='Mouth')
    df2 = pd.DataFrame(metric_propEyes, columns=['Full_base', 'Attention Transfer', 'Impaired']).assign(grp='Eyes')
    df3 = pd.DataFrame(metric_propMouth, columns=['Full_base', 'Attention Transfer', 'Impaired']).assign(grp='Mouth')
    return df0, df1, df2, df3


def get_stats(dfEyes, dfMouth, cols):
    baseEyes, LR01Eyes, LR005Eyes, LR001Eyes, ImparedEyes = dfEyes[cols[0]], dfEyes[cols[1]], dfEyes[cols[2]], dfEyes[cols[3]], dfEyes[cols[4]]
    baseMouth, LR01Mouth, LR005Mouth, LR001Mouth, ImparedMouth = dfMouth[cols[0]], dfMouth[cols[1]], dfMouth[cols[2]], dfMouth[cols[3]], dfMouth[cols[4]]

    _, p1 = stats.ttest_rel(baseEyes, LR01Eyes)
    _, p2 = stats.ttest_rel(baseEyes, LR005Eyes)
    _, p3 = stats.ttest_rel(baseEyes, LR001Eyes)
    _, p4 = stats.ttest_rel(baseEyes, ImparedEyes)

    _, p5 = stats.ttest_rel(baseMouth, LR01Mouth)
    _, p6 = stats.ttest_rel(baseMouth, LR005Mouth)
    _, p7 = stats.ttest_rel(baseMouth, LR001Mouth)
    _, p8 = stats.ttest_rel(baseMouth, ImparedMouth)

    sig = []
    for p in [p1, p2, p3, p4, p5, p6, p7, p8]:
        if  0.01 < p <= 0.05:
            sig.append('*')
        elif 0.001< p <= 0.01:
            sig.append('**')
        elif p <=0.001:
            sig.append('***')
        else:
            sig.append(None)
    return sig


def get_stats_AT(dfEyes, dfMouth, cols):
    baseEyes, ATeyes, ImparedEyes = dfEyes[cols[0]], dfEyes[cols[1]], dfEyes[cols[2]]
    baseMouth, ATmouth, ImparedMouth = dfMouth[cols[0]], dfMouth[cols[1]], dfMouth[cols[2]]

    _, p1 = stats.ttest_rel(baseEyes, ATeyes)
    print(stats.ttest_rel(baseEyes, ATeyes))
    _, p2 = stats.ttest_rel(baseEyes, ImparedEyes)
    print(stats.ttest_rel(baseEyes, ImparedEyes))
    _, p3 = stats.ttest_rel(baseMouth, ATmouth)
    print(stats.ttest_rel(baseMouth, ATmouth))
    _, p4 = stats.ttest_rel(baseMouth, ImparedMouth)
    print(stats.ttest_rel(baseMouth, ImparedMouth))

    sig = []
    for p in [p1, p2, p3, p4]:
        if  0.01 < p <= 0.05:
            sig.append('*')
        elif 0.001< p <= 0.01:
            sig.append('**')
        elif p <=0.001:
            sig.append('***')
        else:
            sig.append(None)
    return sig


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
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # Draw stats
    m = 0.1
    x = 0
    xx = 1
    x1, x2, x3, x4 = x-2*m, x-m, x+m, x+2*m
    xx1, xx2, xx3, xx4 = xx-2*m, xx-m, xx+m, xx+2*m
    y = mdf['value'].max()
    h = 0.02*y

    if test_stats[0]:
        y1 = y+0.1*y
        ax.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=1.5, c='k')
        ax.text((x1+x2)*.5, y1+h, test_stats[0], ha='center', va='bottom', color='k')

    if test_stats[1]:
        y2 = y+0.2*y
        ax.plot([x1, x1, x, x], [y2, y2+h, y2+h, y2], lw=1.5, c='k')
        ax.text((x1+x)*.5, y2+h, test_stats[1], ha='center', va='bottom', color='k')

    if test_stats[2]:
        y3 = y+0.3*y
        ax.plot([x1, x1, x3, x3], [y3, y3+h, y3+h, y3], lw=1.5, c='k')
        ax.text((x1+x3)*.5, y3+h, test_stats[2], ha='center', va='bottom', color='k')

    if test_stats[3]:
        y4 = y+0.4*y
        ax.plot([x1, x1, x4, x4], [y4, y4+h, y4+h, y4], lw=1.5, c='k')
        ax.text((x1+x4)*.5, y4+h, test_stats[3], ha='center', va='bottom', color='k')

    if test_stats[4]:
        y1 = y+0.1*y
        ax.plot([xx1, xx1, xx2, xx2], [y1, y1+h, y1+h, y1], lw=1.5, c='k')
        ax.text((xx1+xx2)*.5, y1+h, test_stats[0], ha='center', va='bottom', color='k')

    if test_stats[5]:
        y2 = y+0.2*y
        ax.plot([xx1, xx1, xx, xx], [y2, y2+h, y2+h, y2], lw=1.5, c='k')
        ax.text((xx1+xx)*.5, y2+h, test_stats[1], ha='center', va='bottom', color='k')

    if test_stats[6]:
        y3 = y+0.3*y
        ax.plot([xx1, xx1, xx3, xx3], [y3, y3+h, y3+h, y3], lw=1.5, c='k')
        ax.text((xx1+xx3)*.5, y3+h, test_stats[2], ha='center', va='bottom', color='k')

    if test_stats[7]:
        y4 = y+0.4*y
        ax.plot([xx1, xx1, xx4, xx4], [y4, y4+h, y4+h, y4], lw=1.5, c='k')
        ax.text((xx1+xx4)*.5, y4+h, test_stats[3], ha='center', va='bottom', color='k')


def draw_boxplot_AT(df1, df2, y_name, pal, test_stats):
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
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # Draw stats
    m = 0.16
    x = 0
    xx = 1
    x1, x2 = x-m, x+m
    xx1, xx2 = xx-m, xx+m
    y = mdf['value'].max()
    h = 0.02*y

    if test_stats[0]:
        y1 = y+0.1*y
        ax.plot([x1, x1, x, x], [y1, y1+h, y1+h, y1], lw=1.5, c='k')
        ax.text((x1+x)*.5, y1+h, test_stats[0], ha='center', va='bottom', color='k')

    if test_stats[1]:
        y2 = y+0.2*y
        ax.plot([x1, x1, x2, x2], [y2, y2+h, y2+h, y2], lw=1.5, c='k')
        ax.text((x1+x2)*.5, y2+h, test_stats[1], ha='center', va='bottom', color='k')

    if test_stats[2]:
        y1 = y+0.1*y
        ax.plot([xx1, xx1, xx, xx], [y1, y1+h, y1+h, y1], lw=1.5, c='k')
        ax.text((xx1+xx)*.5, y1+h, test_stats[0], ha='center', va='bottom', color='k')

    if test_stats[3]:
        y2 = y+0.2*y
        ax.plot([xx1, xx1, xx2, xx2], [y2, y2+h, y2+h, y2], lw=1.5, c='k')
        ax.text((xx1+xx2)*.5, y2+h, test_stats[1], ha='center', va='bottom', color='k')


def get_metric(path):
    with open(path, 'rb') as f:
        metric =pkl.load(f)
    Avg_eyes = metric[:,0]
    Avg_mouth = metric[:,1]
    Prop_eyes = metric[:,2]
    Prop_mouth = metric[:,3]
    print('Eyes mean: %.2f std: %.2f' % (Avg_eyes.mean(), Avg_eyes.std()), 
        'Mouth mean: %.2f std: %.2f' % (Avg_mouth.mean(), Avg_mouth.std()),
        'Eyes prop: %.2f std: %.2f' % (Prop_eyes.mean(), Prop_eyes.std()),
        'Mouth prop: %.2f std: %.2f' % (Prop_mouth.mean(), Prop_mouth.std()))
    return [Avg_eyes, Avg_mouth, Prop_eyes, Prop_mouth]


save_dir = './result_new_boxplot/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
metric_name = ['Avg_eyes', 'Avg_mouth', 'Prop_eyes', 'Prop_mouth']
path_base = './stats_final/gradcam/full_baseModel_metrics.pkl'
metric_base = get_metric(path_base)
path_80 = './stats_final/gradcam/full_on_mouth80_metrics.pkl'
metric_80 = get_metric(path_80)

cols=['Full_base', 'LR=0.01', 'LR=0.005', 'LR=0.001', 'Impaired']

# Recovery by LR
path_LR01 = './stats_final/gradcam_recoveryLR/recoveryLR_0.01_metrics.pkl'
metric_LR01 = get_metric(path_LR01)
path_LR005 = './stats_final/gradcam_recoveryLR/recoveryLR_0.005_metrics.pkl'
metric_LR005 = get_metric(path_LR005)
path_LR001 = './stats_final/gradcam_recoveryLR/recoveryLR_0.001_metrics.pkl'
metric_LR001 = get_metric(path_LR001)

my_pal = ['#E31A1C', '#009392', '#39B1B5', '#9CCB86', '#666666']

df0, df1, df2, df3 = get_df(path_base, path_LR01, path_LR005, path_LR001, path_80)
sig_avg = get_stats(df0, df1, cols)
sig_prop = get_stats(df2, df3, cols)

draw_boxplot(df0, df1, 'Average Intensity', my_pal, test_stats=sig_avg)
name_avg = 'fullFace_on_mouthModel_recoveryLR_Avg'
plt.title(name_avg)
plt.savefig(save_dir + f'{name_avg}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.svg', bbox_inches='tight', dpi=100)

draw_boxplot(df2, df3, 'Intensity Proportion', my_pal, test_stats=sig_prop)
name_prop = 'fullFace_on_mouthModel_recoveryLR_Prop'
plt.title(name_prop)
plt.savefig(save_dir + f'{name_prop}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.svg', bbox_inches='tight', dpi=100)

stats.ttest_rel(metric_LR01[0], metric_base[0])
stats.ttest_rel(metric_LR01[2], metric_base[2])

# Recovery by AT
cols=['Full_base', 'Attention Transfer', 'Impaired']
path_AT = './stats_final/gradcam_recovery_AT/recovery_5_1_2_metrics.pkl'
metric_AT = get_metric(path_AT)

my_pal = ['#E31A1C', '#045275', '#666666']

df0, df1, df2, df3 = get_df_AT(path_base, path_AT, path_80)
sig_avg = get_stats_AT(df0, df1, cols)
sig_prop = get_stats_AT(df2, df3, cols)

draw_boxplot_AT(df0, df1, 'Average Intensity', my_pal, test_stats=sig_avg)
name_avg = 'fullFace_on_mouthModel_recoveryAT_Avg'
plt.title(name_avg)
plt.savefig(save_dir + f'{name_avg}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_avg}.svg', bbox_inches='tight', dpi=100)
draw_boxplot_AT(df2, df3, 'Intensity Proportion', my_pal, test_stats=sig_prop)
name_prop = 'fullFace_on_mouthModel_recoveryAT_Prop'
plt.title(name_prop)
plt.savefig(save_dir + f'{name_prop}.png', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.eps', bbox_inches='tight', dpi=100)
plt.savefig(save_dir + f'{name_prop}.svg', bbox_inches='tight', dpi=100)

stats.ttest_rel(metric_80[0], metric_AT[0])
stats.ttest_rel(metric_80[2], metric_AT[2])

stats.ttest_rel(metric_LR001[0], metric_AT[0])
stats.ttest_rel(metric_LR001[2], metric_AT[2])


# def get_df(path_base, path01, path005, path001, path_80, path_AT):
#     with open(path_base, 'rb') as f:
#         metric_base = pkl.load(f)
#     with open(path01, 'rb') as f:
#         metric01 = pkl.load(f)
#     with open(path005, 'rb') as f:
#         metric005 = pkl.load(f)
#     with open(path001, 'rb') as f:
#         metric001 = pkl.load(f)
#     with open(path_80, 'rb') as f:
#         metric_80 = pkl.load(f)
#     with open(path_AT, 'rb') as f:
#         metric_AT = pkl.load(f)

#     metric_avgEyes = np.array(
#         [metric_base[:, 0], metric01[:, 0], metric005[:, 0], metric001[:, 0], metric_80[:, 0], metric_AT[:, 0]]
#         ).transpose()
#     metric_avgMouth = np.array(
#         [metric_base[:, 1], metric01[:, 1], metric005[:, 1], metric001[:, 1], metric_80[:, 1], metric_AT[:, 1]]
#         ).transpose()
#     metric_propEyes = np.array(
#         [metric_base[:, 2], metric01[:, 2], metric005[:, 2], metric001[:, 2], metric_80[:, 2], metric_AT[:, 2]]
#         ).transpose()
#     metric_propMouth = np.array(
#         [metric_base[:, 3], metric01[:, 3], metric005[:, 3], metric001[:, 3], metric_80[:, 3], metric_AT[:, 3]]
#         ).transpose()

#     df0 = pd.DataFrame(metric_avgEyes, columns=['Full_base', 'LR=0.01', 'LR=0.005', 'LR=0.001', 'Impaired', 'AT']).assign(grp='Eyes')
#     df1 = pd.DataFrame(metric_avgMouth, columns=['Full_base', 'LR=0.01', 'LR=0.005', 'LR=0.001', 'Impaired', 'AT']).assign(grp='Mouth')
#     df2 = pd.DataFrame(metric_propEyes, columns=['Full_base', 'LR=0.01', 'LR=0.005', 'LR=0.001', 'Impaired', 'AT']).assign(grp='Eyes')
#     df3 = pd.DataFrame(metric_propMouth, columns=['Full_base', 'LR=0.01', 'LR=0.005', 'LR=0.001', 'Impaired', 'AT']).assign(grp='Mouth')
#     return df0, df1, df2, df3

# my_pal = ['#E31A1C', '#009392', '#39B1B5', '#9CCB86', '#666666', '#045275']
# df0, df1, df2, df3 = get_df(path_base, path_LR01, path_LR005, path_LR001, path_80, path_AT)
# draw_boxplot(df0, df1, 'Average Intensity', my_pal)
# draw_boxplot(df2, df3, 'Intensity Proportion', my_pal)

