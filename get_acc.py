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

def get_df(log_dir):
    df = tflog2pandas(log_dir)
    df = df[(df.metric) == 'acc/valid']
    df = df['value'] 
    return df

def print_acc(log_dir):
    print(log_dir.split('/')[-1] + ' :', max(get_df(log_dir)))
    
log_root = './logs/CASIA_WebFace_20000_0.15_final'
log_dir_list = sorted([os.path.join(log_root, f) for f in os.listdir(log_root)])
for log_dir in log_dir_list:
    print_acc(log_dir)

log_dir = './logs/CASIA_WebFace_20000_0.15_final_recovery_v1/[resnet50]_[4]_[1]_[10]_[mouth0.15_80]'
print_acc(log_dir)