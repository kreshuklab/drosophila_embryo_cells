import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.morphology import distance_transform_edt
from scipy.stats import linregress
from skimage.future import graph
from sklearn.linear_model import LinearRegression


def get_myo_around(idx, tf, n=10):
    no_cell_mask = segmentation[tf] != idx
    dist_tr = distance_transform_edt(no_cell_mask)
    mask_around = (dist_tr <= n) * no_cell_mask
    myo_around = myosin[tf] * mask_around
    return np.sum(myo_around) / np.sum(mask_around) * 0.0148


def get_myo_in(idx, tf):
    cell_mask = segmentation[tf] == idx
    myo_in = myosin[tf] * cell_mask
    return np.sum(myo_in) / np.sum(cell_mask) * 0.0148


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
        idx_data = idx_data[idx_data['frame_nb'] > 2]
        row = idx_data['row_id'].unique()[0]
        tps, myo, area = [np.array(idx_data[k])
                          for k in ['frame_nb', 'concentration_myo', 'area_cells']]
        if len(tps) < 5: continue
        #myo = [get_myo_in(idx, tp) for tp in tps]
        myo = smooth(myo, sigma=myo_s, tolerance=1)
        area = smooth(area, sigma=area_s, tolerance=0.1)
        all_myo_conc[idx] = {t: m for t, m in zip(tps[1:-1], myo)}
        all_sizes[idx] = {t: s for t, s in zip(tps[1:-1], area)}
        idx2row[idx] = row
    return all_myo_conc, all_sizes, idx2row


def get_myo_time_points(myo_conc, sizes, row_nums, myo_tp=3):
    points_list = []
    for idx in myo_conc.keys():
        tps = myo_conc[idx].keys()
        for tp in range(min(tps) - 1 + myo_tp, max(tps) - 1):
            if idx not in segmentation[tp] or idx not in segmentation[tp + 1]: continue
            size_change = sizes[idx][tp + 1] / sizes[idx][tp]
            cell_myo = myo_conc[idx][tp]
            nbr_myo = np.mean([get_myo_around(idx, i, n=70) for i in range(tp, tp - myo_tp, -1)])
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
myo, area, rows = get_size_and_myo_dict(data_table, myo_s=3, area_s=3)
to_plot = get_myo_time_points(myo, area, rows, 1)
get_best_regr(to_plot, 400)

## the loglog plot
plt.scatter(to_plot[:, 1], to_plot[:, 2], c=to_plot[:, 0], cmap='RdYlGn', vmin=0.9, vmax=1.1)
plt.xlabel("Cell's myosin concentration (log)", size=25)
plt.ylabel("Myosin concentration in the neighborhood (log)", size=25)
plt.loglog()
plt.colorbar()
plt.show()


# the zoom in plot oclored by rows
color_dict = {3:'magenta', 4:'indigo', 5:'skyblue', 6:'mediumseagreen', 7:'red', 8:'orange'}
plot_cutout = to_plot[(3 < to_plot[:, 1]) & (to_plot[:, 1] < 4)]
slope, intercept, rvalue, _, _ = linregress(np.log(plot_cutout[:, 2]), plot_cutout[:, 0])
y = intercept + slope * np.log(plot_cutout[:, 2])
plt.plot(np.log(plot_cutout[:, 2]), y, 'red', label='linear fit')
for row in np.unique(to_plot[:, 3]):
    row_data = plot_cutout[plot_cutout[:, 3] == row]
    plt.scatter(np.log(row_data[:, 2]), row_data[:, 0], c=color_dict[row],
                label="Row {}".format(int(row)))

plt.ylabel("Relative size change", size=25)
plt.xlabel("Myosin concentration in the neighborhood (log)", size=25)
plt.text(3.3, 0.94, "Correlation=0.723", size=20)
plt.legend(loc='upper left', fontsize=15)
plt.show()

# the loglog plot for each row separately
for i in np.unique(to_plot[:, 3]):
    row_data = to_plot[to_plot[:, 3] == i]
    plt.scatter(row_data[:, 1], row_data[:, 2], c=row_data[:, 0], cmap='RdYlGn', vmin=0.9, vmax=1.1)
    plt.xlabel("Cell's myosin concentration (log)", size=25)
    plt.ylabel("Myosin concentration in the neighborhood (log)", size=25)
    plt.title("Row {}".format(int(i)), size=30)
    plt.loglog()
    plt.colorbar()
    plt.show()
