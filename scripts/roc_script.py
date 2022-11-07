import math
# import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import sys

suffix = 'Pre-cores=1-local'
suffix = 'prefilter-cores=1-cluster'

if len(sys.argv) >= 2:
    print(sys.argv)
    print("before:", suffix)
    suffix = sys.argv[1]
print("suffix is now:", suffix)

basepath = '/home/mpg01/MBPC/m.faghihkhorasani/dbtmp2'
qhpath = f'{basepath}/queryDB{suffix}_h'
thpath = f'{basepath}/targetDB{suffix}_h'
# scorepath = f'{basepath}/resDB{suffix}.tsv'
scorepath = f'{basepath}/Pre-cores=1-local_1x3374007.tsv'
scorepath = f'{basepath}/resDBprefilter-cores=1-cluster.tsv'
scorepath = '/home/mpg01/MBPC/m.faghihkhorasani/dbtmp/resDB100_e6_prefilter8-8-cluster.tsv'

print(scorepath, qhpath, thpath)
def create_header_index(path, verbose=False):
    res = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip().replace('\x00', '')
            if verbose:
                print(line, len(line))
            if not len(line) > 0:
                continue
            id_, *fams = line.split("|")[0].split()
            res[id_] = fams
    return res
# print(create_header_index('bulk_tmpheader'))

def read_scores(path, qh, th):
    # 0: common family sequences, 1: not common family but common class sequences, 2: no common family and class
    res = {0: [], 1: [], 2: []}
    with open(path, 'r') as f:
        all_cnt = 0; cnt1 = 0; cnt2 = 0
        for line in f:
            line = line.rstrip().replace('\x00', '')
            q, t, sc, diag = map(str, line.split())
            qs = qh[q]
            ts = th[t]
            score = int(sc)
            if all_cnt % 100 == 0:
                print(qs, ts)
            if set(qs).isdisjoint(ts):
                qs2 = list(map(lambda x: x.split(".")[0], qs))
                ts2 = list(map(lambda x: x.split(".")[0], ts))
                if set(qs2).isdisjoint(ts2):
                    cnt2 += 1
                    res[2].append(score)
                else:
                    res[1].append(score)
                cnt1 += 1
            else:
                res[0].append(score)
            all_cnt += 1
    return res, all_cnt, cnt1, cnt2

qh = create_header_index(qhpath)
th = create_header_index(thpath)
# print(qh)
new_res = read_scores(scorepath, qh, th)
print(new_res[1:])

mx = int(1e4)
scale_div = math.log10(mx)
def log_scale(score):
    # >=10^5 -> 1, 1 -> 0
    return math.log10(min(mx, score)) / scale_div

def lin_scale(score):
    return min(mx, score) / mx

def log_scale_inv(value):
    return 10 ** (scale_div * value)

def lin_scale_inv(value):
    return value * mx

def rates(trues, falses, thr):
    tpr = len(trues[trues > thr]) / len(trues)
    try:
        fpr = len(falses[falses > thr]) / len(falses)
    except:
        fpr = 0.0
    return tpr, fpr, thr

def roc_plot(dict_data, map_func=log_scale, thr_num=50000, save_fig=True, save_path='roc.png'):
    thresholds = np.linspace(0, 1, 50000)
    tmp = copy.deepcopy(dict_data)
    tmp[1] += tmp[2]

    trues = np.array([map_func(score) for score in tmp[0]])
    falses = np.array([map_func(score) for score in tmp[1]])

    points = []
    for thr in thresholds:
        tmp1 = rates(trues, falses, thr)
        points.append(tmp1)
    pivot = pd.DataFrame(points, columns=["y", "x", 'threshold'])
    plt.plot(pivot.x, pivot.y)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if save_fig:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    auc_score = round(abs(np.trapz(pivot.y, pivot.x)), 4)
    return pivot, auc_score, points

trues, falses = np.array(new_res[0][0]), np.array(new_res[0][1])
# for i in range(40, 66):
#     print(rates(trues, falses, i))
pivot, auc_score, points = roc_plot(new_res[0], log_scale)
print(auc_score)
