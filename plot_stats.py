import os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


root = './stats_final/gradcam_recovery/'
x1, x2 = 0, 1
col = 'k'
save_dir = './result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# path = './stats_final/gradcam/full_baseModel_metrics.pkl'
path_list = sorted([os.path.join(root, f) for f in os.listdir(root) if 'metric' in f])
for path in path_list:
    name = path.split('/')[-1].split('.')[0]
    print(name)
    with open(path, 'rb') as f:
        metric = pkl.load(f)
    metric_name = ['Avg_eyes', 'Avg_mouth', 'Prop_eyes', 'Prop_mouth']
    df = pd.DataFrame(metric, columns=metric_name)
    mena_Avg_eyes = df['Avg_eyes'].mean()
    mena_Avg_mouth = df['Avg_mouth'].mean()
    mena_Prop_eyes = df['Prop_eyes'].mean()
    mena_Prop_mouth = df['Prop_mouth'].mean()

    print('%.2f' % mena_Avg_eyes, '%.2f' % mena_Avg_mouth)
    print('%.2f' % mena_Prop_eyes,'%.2f' % mena_Prop_mouth)
    t1, p1 = stats.ttest_ind(df['Avg_eyes'], df['Avg_mouth'])
    print(t1, p1)
    t2, p2 = stats.ttest_ind(df['Prop_eyes'], df['Prop_mouth'])
    print(t2, p2)

    # fig, axes = plt.subplots(1, 2)
    # sns.boxplot(
    #     data=df[['Avg_eyes', 'Avg_mouth']], width=0.5,
    #     showmeans=True, 
    #     meanprops={'marker':'*','markerfacecolor':'white', 'markeredgecolor':'black'},
    #     ax=axes[0]
    # )
    
    # axes[0].set_xticklabels(['eyes', 'mouth'], rotation=0)
    # axes[0].set(ylabel='Average Intensity')
    # axes[0].grid(alpha=0.5)

    # y = max(df[metric_name[0]].max(), df[metric_name[1]].max())
    # y = y + 0.1*y
    # h = 0.02*y
    # axes[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    # axes[0].text((x1+x2)*.5, y+h, '***', ha='center', va='bottom', color=col)
    # # plt.savefig(save_dir + f'{name}_avg.png', bbox_inches='tight', dpi=100)

    # sns.boxplot(
    #     data=df[['Prop_eyes', 'Prop_mouth']], showmeans=True, width=0.5,
    #     meanprops={'marker':'*','markerfacecolor':'white', 'markeredgecolor':'black'},
    #     ax=axes[1]
    # )
    # axes[1].set_xticklabels(['eyes', 'mouth'], rotation=0)
    # axes[1].set(ylabel='Intensity Proportion')
    # axes[1].yaxis.set_label_position("right")
    # axes[1].yaxis.tick_right()
    # axes[1].grid(alpha=0.5)

    # y = max(df[metric_name[2]].max(), df[metric_name[3]].max())
    # y = y + 0.1*y
    # h = 0.02*y
    # axes[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    # axes[1].text((x1+x2)*.5, y+h, '***', ha='center', va='bottom', color=col)
    # # plt.savefig(save_dir + f'{name}_prop.png', bbox_inches='tight', dpi=100)
    # plt.subplots_adjust(wspace=0.01, hspace=0.1)
    # plt.suptitle(name)
    # plt.savefig(save_dir + f'{name}.png', bbox_inches='tight', dpi=100)