import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.morphology import distance_transform_edt
from scipy.stats import linregress
from skimage.future import graph
from skimage.measure import regionprops
from sklearn.linear_model import LinearRegression


def get_myo_around(idx, tf, n=10, exclude=None, cut=None):
    no_cell_mask = segmentation[tf] != idx
    dist_tr = distance_transform_edt(no_cell_mask)
    mask_around = (dist_tr <= n) * no_cell_mask
    if exclude is not None:
        assert cut is not None
        myo_around = cut_doughnut(mask_around, np.invert(no_cell_mask), cut, exclude)
    myo_around = myosin[tf] * mask_around
    return np.sum(myo_around) / (np.sum(mask_around) * 0.0148)


def show_myo(idx, tf, n=70):
    no_cell_mask = segmentation[tf] != idx
    cell_mask = segmentation[tf] == idx
    dist_tr = distance_transform_edt(no_cell_mask)
    cell_countour = (dist_tr <= 2) * no_cell_mask
    myo_countour = (dist_tr < n+1) * (dist_tr > n-1)
    mask_around = (dist_tr <= n) * no_cell_mask
    myo_around = myosin[tf] * mask_around
    myo_in = myosin[tf] * cell_mask
    viewer = napari.Viewer()
    viewer.add_image(cell_countour + myo_countour, blending='additive')
    viewer.add_image(myo_around + myo_in, blending='additive')


def cut_doughnut(myo_mask, cell_mask, line='h', excl='in'):
    x_min, y_min, x_max, y_max = regionprops(cell_mask.astype(int))[0]['bbox']
    if line == 'h' and excl == 'in':
        myo_mask[x_min:x_max] = 0
    if line == 'h' and excl == 'out':
        myo_mask[:x_min] = 0
        myo_mask[x_max:] = 0
    if line == 'v' and excl == 'in':
        myo_mask[:, y_min:y_max] = 0
    if line == 'v' and excl == 'out':
        myo_mask[:, :y_min] = 0
        myo_mask[:, y_max:] = 0
    return myo_mask


def get_myo_in(idx, tf):
    cell_mask = segmentation[tf] == idx
    myo_in = myosin[tf] * cell_mask
    return np.sum(myo_in) / (np.sum(cell_mask) * 0.0148)


def get_area(idx, tf):
    return np.sum(segmentation[tf] == idx)


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


def get_size_and_myo_dict(myo_s=3, area_s=3):
    all_myo_conc = {}
    all_sizes = {}
    idx2row = {}
    for idx in np.unique(segmentation):
        if idx == 0: continue
        tps = [tp for tp, segm_tp in enumerate(segmentation) if (idx in segm_tp)]
        if len(tps) < 5: continue
        myo = [get_myo_in(idx, tp) for tp in tps]
        myo = smooth(myo, sigma=myo_s, tolerance=1)
        area = [get_area(idx, tp) for tp in tps]
        area = smooth(area, sigma=area_s, tolerance=0.1)
        all_myo_conc[idx] = {t: m for t, m in zip(tps[1:-1], myo)}
        all_sizes[idx] = {t: s for t, s in zip(tps[1:-1], area)}
    return all_myo_conc, all_sizes


def get_myo_time_points(myo_conc, sizes, offs, ex=None, plane=None):
    points_list = []
    for idx in myo_conc.keys():
        tps = myo_conc[idx].keys()
        for tp in range(min(tps), max(tps) - 1):
            if tp not in tps or tp+1 not in tps: continue
            size_change = sizes[idx][tp + 1] / sizes[idx][tp]
            cell_myo = myo_conc[idx][tp]
            nbr_myo = get_myo_around(idx, tp, 70, ex, plane)
            points_list.append([size_change, cell_myo, nbr_myo, idx, tp])
    return np.array(points_list)


def train_regr(data):
    np.random.shuffle(data)
    half = int(len(data) / 2)
    data, labels = data[:, 1:3], data[:, 0]
    linear_regr = LinearRegression(normalize=True)
    linear_regr.fit(data[:half], labels[:half])
    score = linear_regr.score(data[half:], labels[half:])
    return score


def get_best_regr(data, n=100):
    accuracies = [train_regr(data) for i in range(n)]
    print("Max accuracy is", np.max(accuracies))
    print("Mean accuracy is", np.mean(accuracies))


data_h5 = '/home/zinchenk/work/drosophila_emryo_cells/data/img5_new.h5'

with h5py.File(data_h5, 'r') as f:
    myosin = f['myosin'][3:-3]
    segmentation = f['segmentation'][3:-3]

myo, area = get_size_and_myo_dict(myo_s=3, area_s=3)
to_plot = get_myo_time_points(myo, area)
get_best_regr(to_plot, 400)


## the loglog plot
fig, ax = plt.subplots()
plt.scatter(to_plot[:, 1], to_plot[:, 2], c=to_plot[:, 0], cmap='RdYlBu', vmin=0.9, vmax=1.1, s=20)
#plt.plot([1,180], [1,180], c='black', linewidth=0.5)
ax.vlines([80000, 100000], 24000, 220000, linestyles='dotted')
ax.hlines([24000, 220000], 80000, 100000, linestyles='dotted')
plt.xlabel("[cellular myosin]", size=35)
plt.ylabel("[surrounding myosin]", size=35)
#plt.title('Embryo 5', size=35)
[tick.label.set_fontsize(25) for tick in ax.xaxis.get_major_ticks()]
[tick.label.set_fontsize(25) for tick in ax.yaxis.get_major_ticks()]
#plt.xlim(0.9, 300)
#plt.ylim(0.7, 190)
plt.loglog()
cb = plt.colorbar()
for t in cb.ax.get_yticklabels():
     t.set_fontsize(25)

plt.show()


# the zoom in plot colored by size
plot_cutout = to_plot[(80000 < to_plot[:, 1]) & (to_plot[:, 1] < 100000)]
slope, intercept, rvalue, _, _ = linregress(plot_cutout[:, 0], np.log(plot_cutout[:, 2]))
y = intercept + slope * plot_cutout[:, 0]
fig, ax = plt.subplots()
ax.plot(plot_cutout[:, 0], y, 'red', label='linear fit')
#ax.scatter(plot_cutout[:, 0], np.log(plot_cutout[:, 2]), s=160, c=plot_cutout[:, 0], cmap='RdYlBu')
ax.scatter(plot_cutout[:, 0], np.log(plot_cutout[:, 2]), s=200, c='tab:grey')
plt.xlabel("Relative size change", size=35)
plt.ylabel("[surrounding myosin]", size=35)
plt.text(1.04, 10.5, "Correlation={:.4f}".format(rvalue), size=35)
plt.legend(loc='upper left', fontsize=25)
[tick.label.set_fontsize(25) for tick in ax.xaxis.get_major_ticks()]
[tick.label.set_fontsize(25) for tick in ax.yaxis.get_major_ticks()]
plt.show()


# the ratio vs size change plot
exp = to_plot[np.where(to_plot[:, 0] > 1.015)]
constr = to_plot[np.where(to_plot[:, 0] < 0.985)]
middle = to_plot[np.where((to_plot[:, 0] >= 0.985) & (to_plot[:, 0] <= 1.015))]
fig, ax = plt.subplots()
ax.scatter(exp[:, 1] / exp[:, 2], exp[:, 0], c='tab:blue')
ax.scatter(constr[:, 1] / constr[:, 2], constr[:, 0], c='tab:red')
ax.scatter(middle[:, 1] / middle[:, 2], middle[:, 0], c='y')
ax.hlines(1, 0.4, 4.9, color='black')
ax.vlines(1, 0.83, 1.10, color='black')
[tick.label.set_fontsize(25) for tick in ax.xaxis.get_major_ticks()]
[tick.label.set_fontsize(25) for tick in ax.yaxis.get_major_ticks()]
plt.xlabel("cellular/neighbourhood myosin ratio", size=35)
plt.ylabel("relative size change", size=35)
#plt.title('Embryo 5', size=35)
#plt.legend(loc='lower right', fontsize=25)
plt.show()


sm_range = np.arange(0.25, 5.25, 0.125)
fig, ax = plt.subplots()
plt.hist(exp[:, 1] / exp[:, 2], bins=sm_range, density=True, histtype='bar', label='Expanding', color='tab:blue', alpha=0.6)
plt.hist(constr[:, 1] / constr[:, 2], bins=sm_range, density=True, histtype='bar', label='Constricting', color='tab:red', alpha=0.6)
plt.ylabel('probability density', size=35)
plt.xlabel('cellular/neighbourhood myosin ratio', size=35)
plt.legend(loc='upper right', fontsize=25)
[tick.label.set_fontsize(25) for tick in ax.xaxis.get_major_ticks()]
[tick.label.set_fontsize(25) for tick in ax.yaxis.get_major_ticks()]
#plt.title('Embryo 5', size=35)
plt.show()
