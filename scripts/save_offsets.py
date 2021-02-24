import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt


def get_myo_offset(idx, tf, n=70, pts=10):
    intervals = np.arange(0, n + 1, int(n / pts))
    no_cell_mask = segmentation[tf] != idx
    dist_tr = distance_transform_edt(no_cell_mask)
    concentrations = []
    for i in range(pts):
        start, stop = intervals[i], intervals[i+1]
        mask_around = (start < dist_tr) & (dist_tr <= stop) * no_cell_mask
        myo_around = myosin[tf] * mask_around
        concentrations.append(np.sum(myo_around) / (np.sum(mask_around)))
    concentrations = np.array(concentrations)
    return np.mean([i * c for i, c in enumerate(concentrations)]) / np.sum(concentrations)


def get_myo_offset1(idx, tf, n=70):
    no_cell_mask = segmentation[tf] != idx
    dist_tr = distance_transform_edt(no_cell_mask)
    dist_tr_around = dist_tr * (dist_tr <= n) * no_cell_mask
    mask_around = dist_tr_around > 0
    myo_around = myosin[tf] * mask_around
    weighed_myo = myosin[tf] * dist_tr_around
    return np.sum(weighed_myo) / np.sum(myo_around)


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


data_h5 = '/home/zinchenk/work/drosophila_emryo_cells/data/img5.h5'
tracking_csv = '/home/zinchenk/work/drosophila_emryo_cells/data/img5.csv'
PIXEL_SIZE = 0.1217

with h5py.File(data_h5, 'r') as f:
    myosin = f['myosin'][:]
    segmentation = f['segmentation'][:]

data_table = pd.read_csv(tracking_csv)
data_table = remove_border_cells(segmentation, data_table)

offsets = []
for i, row in data_table.iterrows():
    offsets.append(get_myo_offset1(row['new_id'], row['frame_nb']) * PIXEL_SIZE)

data_table['sur_offsets'] = offsets

colors = ('magenta','indigo','skyblue','mediumseagreen','red','orange')
df = data_table[['frame_nb', 'sur_offsets', 'row_id']]
fig, ax = plt.subplots()
for row, c in zip(np.unique(df['row_id']), colors):
    row_df = df[df['row_id'] == row][['frame_nb', 'sur_offsets']]
    df_mean = row_df.groupby('frame_nb').agg(['mean'])
    df_std = row_df.groupby('frame_nb').agg(['std'])
    error_down = ((df_mean.values) - (df_std.values)).flatten()
    error_up = ((df_mean.values) + (df_std.values)).flatten()
    x_axis = range(0, len(df_mean) * 25, 25)
    ax.plot(x_axis, df_mean, color=c, label='row{}'.format(row))
    ax.fill_between(x_axis, error_up, error_down, alpha=0.1, color=c)

[tick.label.set_fontsize(25) for tick in ax.xaxis.get_major_ticks()]
[tick.label.set_fontsize(25) for tick in ax.yaxis.get_major_ticks()]
plt.xlabel("time (s)", size=35)
plt.ylabel("surrounding myosin offset", size=35)
plt.legend(loc='upper left', fontsize=20)
plt.show()
