import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import sys
import glob

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

def read_scores_gpu(path, verbose=False):
    # 0: common family sequences, 1: not common family but common fold sequences, 2: no common family and fold
    res = {0: [], 1: [], 2: []}
    with open(path, 'r') as f:
        all_cnt = 0; cnt1 = 0; cnt2 = 0
        for line in f:
            line = line.rstrip().replace('\x00', '')
            q, t, sc = map(str, line.split())
            qs = q.split('|')[1].split(',')
            ts = t.split('|')[1].split(',')
            score = int(sc)
            if verbose and all_cnt % 2000 == 0:
                print(qs2, ts2)
            if set(qs).isdisjoint(ts):
                qs2 = list(map(lambda x: '_'.join(x.split(".")[:2]), qs))
                ts2 = list(map(lambda x: '_'.join(x.split(".")[:2]), ts))
                if verbose and cnt1 % 1000 == 0:
                    print(qs, ts)
                    print(qs2, ts2)
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

def read_scores_mmseqs(path, qh, th, verbose=False):
    # 0: common family sequences, 1: not common family but common fold sequences, 2: no common family and fold
    res = {0: [], 1: [], 2: []}
    with open(path, 'r') as f:
        all_cnt = 0; cnt1 = 0; cnt2 = 0
        for line in f:
            line = line.rstrip().replace('\x00', '')
            q, t, sc, diag = map(str, line.split())
            qs = qh[q]
            ts = th[t]
            score = int(sc)
            if verbose and all_cnt % 2000 == 0:
                print(qs, ts)
            if set(qs).isdisjoint(ts):
                qs2 = list(map(lambda x: '_'.join(x.split(".")[:2]), qs))
                ts2 = list(map(lambda x: '_'.join(x.split(".")[:2]), ts))
                if verbose and cnt1 % 1000 == 0:
                    print(qs, ts)
                    print(qs2, ts2)
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

def roc_plot(dict_data, map_func=log_scale, thr_num=50000, save_fig=True, save_path='roc.png', label=None, verbose=True):
    thresholds = np.linspace(0, 1, thr_num)
    tmp = copy.deepcopy(dict_data)

    trues = np.array([map_func(score) for score in tmp[0]])
    falses = np.array([map_func(score) for score in tmp[2]])
    # trues = np.array([map_func(score) for score in tmp[0]])
    # falses = np.array([map_func(score) for score in tmp[1] + tmp[2]])
    if verbose:
        print("Counts:", len(trues), len(falses), "=> (Trues [Same Class], Falses [Different Fold])")

    points = []
    for thr in thresholds:
        tmp1 = rates(trues, falses, thr)
        points.append(tmp1)
    pivot = pd.DataFrame(points, columns=["y", "x", 'threshold'])
    plt.plot(pivot.x, pivot.y, label=label)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    if save_fig:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    auc_score = round(abs(np.trapz(pivot.y, pivot.x)), 4)
    return pivot, auc_score, points







suffixU = 'U-cores=8-100x3374007-cluster'
suffixP = 'P-cores=8-100x3374007-cluster'

if len(sys.argv) >= 2:
    print(sys.argv)
    print("before:", suffix)
    suffixP = sys.argv[1]
print("suffixP:", suffixP, "suffixU:", suffixU)

basep = '/home/mpg01/MBPC/m.faghihkhorasani'
basepath = f'{basep}/dbtmp'
qhpath = f'{basepath}/queryDB{suffixP}_h'
thpath = f'{basepath}/targetDB{suffixP}_h'
# scorepath = f'{basepath}/ungapped_{suffixP}.tsv'
mmseqspaths = [(i, "Prefilter" + [f'_{j}' for j in i.split('-') if 'x' in j][0]) for i in glob.glob(f'{basepath}/*{suffixP}.tsv')] \
            + [(i, "UngappedPrefilter" + [f'_{j}' for j in i.split('-') if 'x' in j][0]) for i in glob.glob(f'{basepath}/*{suffixU}.tsv')]
gpupaths = [(i, "GpuPrefilter" + f'_{"x".join(i.split("_")[-2:]) }') for i in glob.glob(f'{basep}/LocalUngappedAlignment/PlayGround_/full_output_2022-10-21_00-38-05_100_3374007')]
print(mmseqspaths, gpupaths, qhpath, thpath, '\n\n')

th = create_header_index(thpath)
qh = create_header_index(qhpath)


for args in [(i + (read_scores_mmseqs, [i[0], qh, th])) for i in mmseqspaths] + [(i + (read_scores_gpu, [i[0]])) for i in gpupaths]:
# for paths in mmseqspaths:
    path = args[0]
    label = args[1]
    print('Datapath:', path, ', Label:', label)
    # new_res = read_scores_mmseqs(path, qh, th)
    new_res = args[2](*args[3])
    print("Counts:", new_res[1:], "=> (ALL, Not Same Class, Not Same Class Nor Same Fold)")

    # trues, falses = np.array(new_res[0][0] + new_res[0][1]), np.array(new_res[0][2])
    trues, falses = np.array(new_res[0][0]), np.array(new_res[0][2])
    # print(len(trues), len(falses))

    pivot, auc_score, points = roc_plot(new_res[0], log_scale, label=label)
    print("AUC:", auc_score)

    ans = [(p[0] - p[1], int(log_scale_inv(p[2])), p[0], p[1]) for p in points]
    ans.sort(reverse=True)
    a = np.array(ans)[:10000]
    res = set(a[(a[:,2] > 0.985)][:,1])
    print("Candidate Thresholds:", res)
    print("TPRs and FPRs in different Candidate Thresholds in the Format of (<TPR>, <FPR>, <THRESOHLD>):")
    for i in sorted(list(res)):
        print(tuple(map(lambda x: round(x, 4), rates(trues, falses, i))))
    print()
