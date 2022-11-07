#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
import shlex
import re
from datetime import timedelta, datetime
import pandas as pd
import glob
import sys


# In[2]:

if len(sys.argv) > 2:
    suffix = sys.argv[1]
else:
    filepath = "~/logtmp2/*cores=8*-*-v=*.log"
search_query = 'Time for processing:'


# In[21]:


files = glob.glob(filepath)
print("filepath:", filepath, '\n', files)
all_runtimes = []
row_labels = []
for file in files:
    row_labels.append('_'.join(file.split('-')[1:]))
    res = subprocess.run(shlex.split(f'grep "{search_query}" {file}'), check=True, text=True, stdout=subprocess.PIPE)
    proc_times = res.stdout.replace(search_query, "").splitlines()
    deltas = []
    print(proc_times)
    total_delta = timedelta(0)
    for t in proc_times:
        match = re.search('(\d*)h (\d*)m (\d*)s (\d*)ms', t)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            milliseconds = int(match.group(4))
            delta = timedelta(
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                milliseconds=milliseconds
            )
            deltas.append(delta)
            total_delta += delta

    deltas.append(total_delta)
    all_runtimes.append([d.total_seconds() for d in deltas])
    labels = ["createdb_query", "createdb_target", "prefilter", "createtsv", "total"]


# In[20]:

df = pd.DataFrame(all_runtimes, columns=labels)
df['label'] = row_labels


# In[23]:


import matplotlib.pyplot as plt

def plot_runtimes(df, save_fig=True, save_path='runtimes_prefilter.png'):
    df.plot.bar(x='label')
    if save_fig:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
plot_runtimes(df)
print('\n', df)

