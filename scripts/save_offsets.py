import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import regionprops
from scipy.ndimage import center_of_mass


def get_myo_offset_in(idx, tf):
    cell_mask = segmentation[tf] == idx
    props = regionprops(cell_mask.astype(int))[0]
    com = np.rint(props.centroid).astype(int)
    com_image = np.ones_like(cell_mask)
    com_image[tuple(com)] = 0
    dist_tr = distance_transform_edt(com_image) * cell_mask
    myo_in = myosin[tf] * cell_mask
    weighed_myo = myo_in * dist_tr
    return np.sum(weighed_myo) / np.sum(myo_in) / props.major_axis_length


def get_myo_offset_diff(idx, tf):
    cell_mask = segmentation[tf] == idx
    com_cell = np.rint(center_of_mass(cell_mask)).astype(int)
    myo_in = myosin[tf] * cell_mask
    com_myo = np.rint(center_of_mass(myo_in)).astype(int)
    props = regionprops(cell_mask.astype(int))[0]
    return np.linalg.norm(com_myo - com_cell)


def get_myo_offset_out(idx, tf, n=70):
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
    offsets.append(get_myo_offset_out(row['new_id'], row['frame_nb']) * PIXEL_SIZE)

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


from skimage.util import map_array
from mpl_toolkits.axes_grid1 import make_axes_locatable

tf = 22
tf_image = segmentation[tf]
props = regionprops(tf_image.astype(int), myosin[tf])
distances = []

plt.figure()
ax = plt.gca()
for p in props:
    _ = plt.scatter(p.centroid[1], p.centroid[0], color='g', s=1000)
    _ = plt.scatter(p.weighted_centroid[1], p.weighted_centroid[0], color='r', s=1000)
    d = [c - w for c, w in zip(p.centroid, p.weighted_centroid)]
    distances.append(np.linalg.norm(d))

value_map = map_array(tf_image, np.unique(tf_image)[1:], np.array(distances))
im = plt.imshow(value_map, vmin=0, vmax=20)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

cb = plt.colorbar(im, cax=cax)
for t in cb.ax.get_yticklabels():
     t.set_fontsize(45)

figure = plt.gcf()
figure.set_size_inches(64, 48)
plt.savefig(fp + 'fig3l_{}.png'.format(tf))
plt.show()

row_map = map_array(segmentation[tf], np.array(data_table.new_id), np.array(data_table.row_id) - 2)
plt.imshow(row_map, cmap=ListedColormap(('white',) + colors))
figure = plt.gcf()
figure.set_size_inches(64, 48)
plt.savefig(fp + 'fig3l_rows_{}.png'.format(tf))
plt.show()
