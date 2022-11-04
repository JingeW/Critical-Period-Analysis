import os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


def get_data(path):
    name = path.split('/')[-1].split('.')[0]
    print(name)
    with open(path, 'rb') as f:
        metric = pkl.load(f)
    metric_name = ['Avg_eyes', 'Avg_mouth', 'Prop_eyes', 'Prop_mouth']
    df = pd.DataFrame(metric, columns=metric_name)
    mean_Avg_eyes = df['Avg_eyes'].mean()
    mean_Avg_mouth = df['Avg_mouth'].mean()
    mean_Prop_eyes = df['Prop_eyes'].mean()
    mean_Prop_mouth = df['Prop_mouth'].mean()
    return df, [mean_Avg_eyes, mean_Avg_mouth, mean_Prop_eyes, mean_Prop_mouth]


root = './stats_final/gradcam_recovery/'
x1, x2 = 0, 1
col = 'k'
save_dir = './result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
path = './stats_final/gradcam/full_on_mouth80_metrics.pkl'
recovery_path = './stats_final/gradcam_recovery/recovery_5_1_2_metrics.pkl'

df, metric = get_data(path)
recovery_df, recovery_metric = get_data(recovery_path)
print('%.2f' % metric[0], '%.2f' % recovery_metric[0])
print('%.2f' % metric[2],'%.2f' % recovery_metric[2])
t1, p1 = stats.ttest_ind(df['Avg_eyes'], recovery_df['Avg_eyes'])
print(t1, p1)
t2, p2 = stats.ttest_ind(df['Prop_eyes'], recovery_df['Prop_eyes'])
print(t2, p2)

data1 = pd.DataFrame(df['Avg_eyes']).assign(Location=1)
data2 = pd.DataFrame(recovery_df['Avg_eyes']).assign(Location=2)
cdf1 = pd.concat([data1, data2])    

data3 = pd.DataFrame(df['Prop_eyes']).assign(Location=1)
data4 = pd.DataFrame(recovery_df['Prop_eyes']).assign(Location=2)
cdf2 = pd.concat([data3, data4])    

fig, axes = plt.subplots(1, 2)
sns.boxplot(
    data=cdf1, x='Location', y='Avg_eyes' ,width=0.5,
    showmeans=True, 
    meanprops={'marker':'*','markerfacecolor':'white', 'markeredgecolor':'black'},
    ax=axes[0]
)
axes[0].set_xticklabels(['Deficit', 'Recovery'], rotation=0)
axes[0].set(ylabel='Average Intensity of Eyes region')
axes[0].grid(alpha=0.5)
axes[0].set(xlabel=None)

y = max(cdf1['Avg_eyes'])
y = y + 0.1*y
h = 0.02*y
axes[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
axes[0].text((x1+x2)*.5, y+h, '***', ha='center', va='bottom', color=col)
# plt.savefig(save_dir + f'{name}_avg.png', bbox_inches='tight', dpi=100)

sns.boxplot(
    data=cdf2, x='Location', y='Prop_eyes' ,width=0.5,
    showmeans=True, 
    meanprops={'marker':'*','markerfacecolor':'white', 'markeredgecolor':'black'},
    ax=axes[1]
)
axes[1].set_xticklabels(['Deficit', 'Recovery'], rotation=0)
axes[1].set(ylabel='Intensity Proportion of Eyes region')
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].grid(alpha=0.5)
axes[1].set(xlabel=None)

y = max(cdf2['Prop_eyes'])
y = y + 0.1*y
h = 0.02*y
axes[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
axes[1].text((x1+x2)*.5, y+h, '***', ha='center', va='bottom', color=col)
# plt.savefig(save_dir + f'{name}_prop.png', bbox_inches='tight', dpi=100)
plt.subplots_adjust(wspace=0.01, hspace=0.1)
plt.suptitle('Mouth80 Recovery')
plt.savefig(save_dir + f'recovery_metric.png', bbox_inches='tight', dpi=100)