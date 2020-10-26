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


def remove_border_cells(segm, data_frame):
    segm_borders = segm.copy()
    segm_borders[:, 2:-2, 2:-2] = 0
    for tf in range(len(segm)):
        border_cells = np.unique(segm_borders[tf])
        border_cells = border_cells[border_cells!=0]
        print(tf, border_cells)
        for cell in border_cells:
            data_frame = data_frame[~((data_frame['frame_nb'] == tf) & (data_frame['new_id'] == cell))]
    return data_frame


def get_myo_around(idx, tf, n=10, exclude=None, cut=None):
    no_cell_mask = segmentation[tf] != idx
    dist_tr = distance_transform_edt(no_cell_mask)
    mask_around = (dist_tr <= n) * no_cell_mask
    if exclude is not None:
        assert cut is not None
        myo_around = cut_doughnut(mask_around, np.invert(no_cell_mask), cut, exclude)
    myo_around = myosin[tf] * mask_around
    return np.sum(myo_around) / np.sum(mask_around) * 0.0148 / 5.955


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
    return np.sum(myo_in) / np.sum(cell_mask) * 0.0148 / 5.955


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


def get_size_and_myo_dict(table, myo_s=3, area_s=3):
    all_myo_conc = {}
    all_sizes = {}
    idx2row = {}
    for idx in table['new_id'].unique():
        #if idx in (79, 81): continue
        idx_data = table[table['new_id'] == idx]
        idx_data = idx_data[idx_data['frame_nb'] >= 2]
        row = idx_data['row_id'].unique()[0]
        tps, myo, area = [np.array(idx_data[k])
                          for k in ['frame_nb', 'concentration_myo', 'area_cells']]
        if len(tps) < 5: continue
        myo = [get_myo_in(idx, tp) for tp in tps]
        myo = smooth(myo, sigma=myo_s, tolerance=1)
        area = smooth(area, sigma=area_s, tolerance=0.1)
        all_myo_conc[idx] = {t: m for t, m in zip(tps[1:-1], myo)}
        all_sizes[idx] = {t: s for t, s in zip(tps[1:-1], area)}
        idx2row[idx] = row
    return all_myo_conc, all_sizes, idx2row


def get_myo_time_points(myo_conc, sizes, row_nums, ex=None, plane=None):
    points_list = []
    for idx in myo_conc.keys():
        tps = myo_conc[idx].keys()
        for tp in range(min(tps), max(tps) - 1):
            if tp not in tps or tp+1 not in tps: continue
            size_change = sizes[idx][tp + 1] / sizes[idx][tp]
            cell_myo = myo_conc[idx][tp]
            nbr_myo = get_myo_around(idx, tp, 70, ex, plane)
            points_list.append([size_change, cell_myo, nbr_myo, row_nums[idx]])
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
data_table = remove_border_cells(segmentation, data_table)
myo, area, rows = get_size_and_myo_dict(data_table, myo_s=3, area_s=3)
to_plot = get_myo_time_points(myo, area, rows)
to_plot = to_plot[~np.isnan(to_plot).any(axis=1)] # WHY DOES this happen?
get_best_regr(to_plot, 400)


## the loglog plot
plt.scatter(to_plot[:, 1], to_plot[:, 2], c=to_plot[:, 0], cmap='RdYlBu', vmin=0.9, vmax=1.1, s=20)
#plt.plot([1,180], [1,180], c='black', linewidth=0.5)
plt.vlines([18/5.955, 22/5.955], 5/5.955, 48/5.955, linestyles='dotted')
plt.hlines([5/5.955, 48/5.955], 18/5.955, 22/5.955, linestyles='dotted')
plt.xlabel("Cell's myosin concentration (log)", size=25)
plt.ylabel("Myosin concentration in the neighborhood (log)", size=25)
#plt.xlim(0.9, 300)
#plt.ylim(0.7, 190)
plt.loglog()
plt.colorbar()
plt.show()


# the zoom in plot colored by rows
color_dict = {3:'magenta', 4:'indigo', 5:'skyblue', 6:'mediumseagreen', 7:'red', 8:'orange'}
plot_cutout = to_plot[(18 < to_plot[:, 1]) & (to_plot[:, 1] < 22)]
slope, intercept, rvalue, _, _ = linregress(np.log(plot_cutout[:, 2]), plot_cutout[:, 0])
y = intercept + slope * np.log(plot_cutout[:, 2])
plt.plot(np.log(plot_cutout[:, 2]), y, 'red', label='linear fit')
for row in np.unique(to_plot[:, 3]):
    row_data = plot_cutout[plot_cutout[:, 3] == row]
    plt.scatter(np.log(row_data[:, 2]), row_data[:, 0], c=color_dict[row],
                label="Row {}".format(int(row)))

plt.ylabel("Relative size change", size=25)
plt.xlabel("Myosin concentration in the neighborhood (log)", size=25)
plt.text(3.3, 0.94, "Correlation=0.7478", size=20)
plt.legend(loc='upper left', fontsize=15)
plt.show()


# the zoom in plot colored by size
plot_cutout = to_plot[(18/5.955 < to_plot[:, 1]) & (to_plot[:, 1] < 22/5.955)]
slope, intercept, rvalue, _, _ = linregress(plot_cutout[:, 0], np.log(plot_cutout[:, 2]))
y = intercept + slope * plot_cutout[:, 0]
fig, ax = plt.subplots()
ax.plot(plot_cutout[:, 0], y, 'red', label='linear fit')
#ax.scatter(plot_cutout[:, 0], np.log(plot_cutout[:, 2]), s=160, c=plot_cutout[:, 0], cmap='RdYlBu')
ax.scatter(plot_cutout[:, 0], np.log(plot_cutout[:, 2]), s=200, c='tab:grey')
plt.xlabel("Relative size change", size=25)
plt.ylabel("Myosin concentration in the neighborhood (log)", size=25)
plt.text(1.06, 0.15, "Correlation=0.7478", size=20)
plt.legend(loc='upper left', fontsize=25)
[tick.label.set_fontsize(15) for tick in ax.xaxis.get_major_ticks()]
[tick.label.set_fontsize(15) for tick in ax.yaxis.get_major_ticks()]
plt.show()



# the loglog plot for each row separately
for i in np.unique(to_plot[:, 3]):
    row_data = to_plot[to_plot[:, 3] == i]
    plt.scatter(row_data[:, 1], row_data[:, 2], c=row_data[:, 0], cmap='RdYlGn', vmin=0.9, vmax=1.1)
    plt.plot([1,180], [1,180], c='black', linewidth=0.5)
    plt.xlabel("Cell's myosin concentration (log)", size=25)
    plt.ylabel("Myosin concentration in the neighborhood (log)", size=25)
    plt.title("Row {}".format(int(i)), size=30)
    plt.xlim(0.9, 300)
    plt.ylim(0.7, 190)
    plt.loglog()
    plt.colorbar()
    plt.show()


# the ratio vs size change plot
exp = to_plot[np.where(to_plot[:, 0] > 1.015)]
constr = to_plot[np.where(to_plot[:, 0] < 0.985)]
middle = to_plot[np.where((to_plot[:, 0] >= 0.985) & (to_plot[:, 0] <= 1.015))]
fig, ax = plt.subplots()
ax.scatter(exp[:, 1] / exp[:, 2], exp[:, 0], c='tab:blue')
ax.scatter(constr[:, 1] / constr[:, 2], constr[:, 0], c='tab:red')
ax.scatter(middle[:, 1] / middle[:, 2], middle[:, 0], c='y')
ax.hlines(1, 0.4, 4.9)
ax.vlines(1, 0.83, 1.10)
[tick.label.set_fontsize(15) for tick in ax.xaxis.get_major_ticks()]
[tick.label.set_fontsize(15) for tick in ax.yaxis.get_major_ticks()]
plt.xlabel("Myosin concentration inside / outside", size=25)
plt.ylabel("Relative size change", size=25)
plt.legend(loc='lower right', fontsize=15)
plt.show()


# the bar plot plot of exp and constr
num_pos, num_mid, num_neg = [], [], []
ratio = to_plot[:, 1] / to_plot[:, 2]
min_value = np.min(ratio)
max_value = np.max(ratio)

values_range = np.arange(0.25, 5.25, 0.25)
for i in range(len(values_range) - 1):
    range_data = to_plot[np.where((ratio > values_range[i]) & (ratio < values_range[i + 1]))]
    num_pos.append(np.sum(range_data[:, 0] > 1.025))
    #num_mid.append(np.sum((range_data[:, 0] < 1.025) & (range_data[:, 0] > 0.95)))
    num_neg.append(np.sum(range_data[:, 0] < 0.975))

width = 0.35
labels = ["{}-{}".format(values_range[i], values_range[i + 1]) for i in range(len(values_range) - 1)]
x = np.arange(len(num_pos))
fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, num_pos, width, label='Expanding', color='tab:blue')
#rects1 = ax.bar(x, num_mid, width, label='Middle')
rects2 = ax.bar(x + width / 2, num_neg, width, label='Constricting', color='tab:red')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Number of cells', size=25)
ax.set_xlabel('Ratio in/out myosin', size=25)
ax.legend(loc='upper left', fontsize=15)
plt.show()

sm_range = np.arange(0.25, 5.25, 0.125)
plt.hist(exp[:, 1] / exp[:, 2], bins=sm_range, density=True, histtype='bar', label='Expanding', color='tab:blue', alpha=0.6)
plt.hist(constr[:, 1] / constr[:, 2], bins=sm_range, density=True, histtype='bar', label='Constricting', color='tab:red', alpha=0.6)
plt.ylabel('Cells density', size=25)
plt.xlabel('Ratio in/out myosin', size=25)
plt.legend(loc='upper left', fontsize=15)
plt.show()
