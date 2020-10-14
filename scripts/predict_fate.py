import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.morphology import distance_transform_edt
from skimage.future import graph
from sklearn.linear_model import LinearRegression


def get_myo_around(idx, tf, n=10):
    no_cell_mask = segmentation[tf] != idx
    dist_tr = distance_transform_edt(no_cell_mask)
    mask_around = (dist_tr <= n) * no_cell_mask
    myo_around = myosin[tf] * mask_around
    return np.sum(myo_around)


def smooth(values, sigma=3, tolerance=0.1):
    values = np.array(values)
    # check if any value is suspicious (definitely a merge)
    for i in range(1, len(values) - 1):
        avg_neigh = (values[i - 1] + values[i + 1]) / 2
        if not (1 + tolerance) > (values[i] / avg_neigh) > (1 - tolerance):
           #replace this value with neighbors' average
           values[i] = avg_neigh
    values = gaussian_filter1d(values, sigma=sigma)
    return values[1:-1]


def get_size_and_myo_dict(table, myo_s=3, area_s=3, row=None):
    all_myo_conc = {}
    all_sizes = {}
    if row is None:
        ids = table['new_id'].unique()
    else:
        ids = table['new_id'][table['row_id'] == row].unique()
    for idx in ids:
        #if idx in (79, 81): continue
        idx_data = table[table['new_id'] == idx]
        idx_data = idx_data[idx_data['frame_nb'] > 2]
        tps, myo, area = [np.array(idx_data[k])
                          for k in ['frame_nb', 'concentration_myo', 'area_cells']]
        if len(tps) < 5: continue
        myo = smooth(myo, sigma=myo_s, tolerance=0.1)
        area = smooth(area, sigma=area_s, tolerance=0.1)
        all_myo_conc[idx] = {t: m for t, m in zip(tps[1:-1], myo)}
        all_sizes[idx] = {t: s for t, s in zip(tps[1:-1], area)}
    return all_myo_conc, all_sizes


def get_myo_time_points(myo_conc, sizes, myo_tp=3):
    points_list = []
    for idx in myo_conc.keys():
        tps = myo_conc[idx].keys()
        for tp in range(min(tps) - 1 + myo_tp, max(tps) - 1):
            if idx not in segmentation[tp] or idx not in segmentation[tp + 1]: continue
            size_change = sizes[idx][tp + 1] / sizes[idx][tp]
            cell_myo = myo_conc[idx][tp]
            nbr_myo = np.max([get_myo_around(idx, tp, n=70) for i in range(tp, tp - myo_tp, -1)])
            points_list.append([size_change, cell_myo, nbr_myo])
    return np.array(points_list)


def train_regr(data):
    np.random.shuffle(data)
    half = int(len(data) / 2)
    data, labels = data[:, 1:], data[:, 0]
    #data = data.reshape(-1, 1)
    linear_regr = LinearRegression(normalize=True)
    linear_regr.fit(data[:half], labels[:half])
    score = linear_regr.score(data[half:], labels[half:])
    return score


def get_best_regr(data, n=100):
    accuracies = [train_regr(data) for i in range(n)]
    print("Max accuracy is", np.max(accuracies))
    print("Mean accuracy is", np.mean(accuracies))


data_h5 = '/home/zinchenk/work/drosophila_emryo_cells/data/img5.h5'
tracking_csv = '/home/zinchenk/work/drosophila_emryo_cells/data/img5.csv'

#data_h5 = '/home/zinchenk/work/drosophila_emryo_cells/data/img24.h5'
#tracking_csv = '/home/zinchenk/work/drosophila_emryo_cells/data/img24.csv'

with h5py.File(data_h5, 'r') as f:
    membranes = f['membranes'][:]
    myosin = f['myosin'][:]
    segmentation = f['segmentation'][:]

data_table = pd.read_csv(tracking_csv)
myo, area = get_size_and_myo_dict(data_table, myo_s=3, area_s=3)
to_plot = get_myo_time_points(myo, area, 1)
get_best_regr(to_plot, 400)

plt.scatter(to_plot[:, 1], to_plot[:, 2], c=to_plot[:, 0], cmap='RdYlGn', vmin=0.9, vmax=1.1)
plt.xlabel("Cell's myo concentration", size=25)
plt.ylabel("Myo amount in the neighborhood", size=25)
plt.colorbar()
plt.show()
