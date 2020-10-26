import h5py
import napari
import numpy as np
import pandas as pd
from vispy.color import Colormap
import matplotlib.pyplot as plt
from skimage.future import graph
from scipy.ndimage import gaussian_filter1d


def make_cdict(label_dict):
    val2color = {val: list(np.random.rand(3)) for val in set(label_dict.values())}
    cdict = {}
    if 0 not in label_dict:
        cdict[0] = [[0., 0., 0., 0.]]
    for idx in sorted(label_dict):
        cdict[idx] = (val2color[label_dict[idx]] + [1])
    return cdict


def smooth_line(values, s=3):
    values = np.array(values)
    #smooth_filter = np.array([0.33,] * 3)
    # to get rid of corner cases
    # check if any value is suspicious (definitely a merge)
    for i in range(1, len(values) - 1):
        avg_neigh = (values[i - 1] + values[i + 1]) / 2
        if (values[i] / avg_neigh) > 1.1 or (values[i] / avg_neigh) < 0.9:
           #replace this value with neighbors' average
           values[i] = avg_neigh
    #values = np.convolve(values, smooth_filter, 'valid')
    values = gaussian_filter1d(values, sigma=s)
    return values[1:-1]


def show_plot(plot, lbl):
    x1, x2, y1, y2 = plot.axis()
    plot.hlines(1, x1, x2, colors='red', linestyles='dashed')
    plot.ylabel(lbl)
    plot.xlabel('Frame number')
    plot.show()


def exp_or_constr(time_points):
    add_filter = np.array([1,] * 3)
    expand = np.any(np.convolve(time_points > 1.2, add_filter, 'valid') == 3)
    constrict = np.any(np.convolve(time_points < 0.8, add_filter, 'valid') == 3)
    print('Constrict: {}, Expand: {}'.format(constrict, expand))


def plot_cells(table, val2plot='area_microns', row_id=None, show_one=False, smooth=True):
    if row_id is None:
        ids = table['new_id'].unique()
    else:
        ids = table['new_id'][table['row_id'] == row_id].unique()
    for idx in ids:
        idx_data = table[[val2plot, 'frame_nb']][table['new_id'] == idx]
        if len(idx_data) < 3:
            continue
        x, y = idx_data['frame_nb'], idx_data[val2plot]
        if smooth:
            y_smooth = smooth_line(y, 3)
            #y_norm_or = y / y.iloc[0]
            y_norm_or = y /  y_smooth[0]
            y_norm_sm = y_smooth / y_smooth[0]
            plt.plot(x[1:-1], y_norm_or[1:-1], color='black')
            plt.plot(x[1:-1], y_norm_sm, color='red')
        else:
            plt.plot(x, y, color=new_cdict[idx][:-1])
        if show_one:
            print(idx)
            #exp_or_constr(y_norm)
            show_plot(plt, val2plot)
    if not show_one:
        show_plot(plt, val2plot)




data_h5 = '/home/zinchenk/work/drosophila_emryo_cells/data/img5.h5'
tracking_csv = '/home/zinchenk/work/drosophila_emryo_cells/data/img5.csv'

data_h5 = '/home/zinchenk/work/drosophila_emryo_cells/data/img24.h5'
tracking_csv = '/home/zinchenk/work/drosophila_emryo_cells/data/img24.csv'



with h5py.File(data_h5, 'r') as f:
    membranes = f['membranes'][:]
    myosin = f['myosin'][:]
    segmentation = f['segmentation'][:]

data_table = pd.read_csv(tracking_csv)

row_ids = {idx: data_table['row_id'][data_table['new_id'] == idx].unique()[0]
          for idx in data_table['new_id'].unique()}
new_cdict = make_cdict(row_ids)

viewer = napari.Viewer()
viewer.add_image(membranes, colormap='green', blending='additive')
viewer.add_image(myosin, colormap='red', blending='additive')
viewer.add_labels(segmentation, blending='additive', color=new_cdict)

plot_cells(data_table, val2plot='area_microns', row_id=7, show_one=True)


def get_data(table, frame_id, value='area_microns'):
    label_dict = get_labels(table)
    lab2cl = {val: i for i, val in enumerate(set(label_dict.values()))}
    frame_table = table[table['frame_nb'] == frame_id]
    data = np.array([[row[value], lab2cl[label_dict[row['new_id']]]]
                       for i, row in frame_table.iterrows()
                      if label_dict[row['new_id']] != 'N'])
    return data


def train_regr(data, types=['C', 'T', 'E']):
    np.random.shuffle(data)
    half = int(len(data) / 2)
    data, labels = data[:, :-1], data[:, -1]
    #data = data.reshape(-1, 1)
    logistic_regr = LogisticRegression(C=0.3)
    logistic_regr.fit(data[:half], labels[:half])
    score = logistic_regr.score(data[half:], labels[half:])
    predictions = logistic_regr.predict(data[half:])
    conf_matrix = metrics.confusion_matrix(labels[half:], predictions)
    print("The accuracy is {0}".format(score))
    print(types)
    print(conf_matrix)

def get_labels(table):
    labels = {idx: table['phenotype'][table['new_id'] == idx].unique()[0]
              for idx in table['new_id'].unique()}
    for idx in table['new_id'][(table['row_id'] != 7) & (table['row_id'] != 8)].unique():
        if labels[idx] == 'E':
            labels[idx] = 'T'
    return labels


def get_myo_conc(idx, tf):
    myosin_tf = myosin[tf]
    segmentation_tf = segmentation[tf]
    cell_mask = segmentation_tf == idx
    cell_size = np.sum(cell_mask)
    myo_sum = np.sum(myosin_tf[cell_mask])
    return myo_sum / cell_size


def get_samestage_cells(myo_conc, tolerance = 0.3):
    for idx in data_table['new_id'].unique():
        idx_data = data_table[data_table['new_id'] == idx]
        closest_entry = idx_data.iloc[np.argmin(np.array(np.abs(idx_data['concentration_myo'] - myo_conc)))]
        tf, conc_myo = closest_entry['frame_nb'], closest_entry['concentration_myo']
        if np.abs(conc_myo - myo_conc) / myo_conc > tolerance: continue
        bb = find_objects(segmentation[tf] == idx)
        viewer = napari.Viewer()
        viewer.add_image(membranes[tf][bb[0]], colormap='green', blending='additive')
        viewer.add_image(myosin[tf][bb[0]], colormap='red', blending='additive')
        #print(closest_entry['row_id'], idx, tf, conc_myo)
        viewer.screenshot('/home/zinchenk/Pictures/group_sem_sept/cell_myo_row{}_tf{}.png'.format(closest_entry['row_id'], tf))
        viewer.close()


def get_avg_conc(idx, table, tf, num_points=3):
    idx_data = table[table['new_id'] == idx]
    conc = [idx_data[idx_data['frame_nb'] == i]['concentration_myo'].item()
            for i in range(tf, tf-num_points, -1)
            if not idx_data[idx_data['frame_nb'] == i].empty]
    return np.mean(conc)


def plot_myo_neighb(table, tf, row, wiggle, expand, constrict, n_pnts=3):
    g = graph.rag_boundary(segmentation[tf], np.ones_like(segmentation[tf], dtype=float))
    ids = table[table['frame_nb'] == tf][table['row_id'] == row]['new_id'].unique()
    for idx in ids:
        if idx in constrict:
            color = 'blue'
        elif idx in wiggle:
            color = 'red'
        elif idx in expand:
            color = 'green'
        else:
            color = 'black'
        myo = get_avg_conc(idx, table, tf, n_pnts)
        myo_neighb = np.mean([get_avg_conc(i, table, tf, n_pnts) for i in g.neighbors(idx) if i!=0])
        plt.scatter(myo, myo_neighb, c=color)
        print(idx, myo, myo_neighb)

def plot_myo_neighb_vs_time(cell_idx):
    g = graph.rag_boundary(segmentation[15], np.ones_like(segmentation[15], dtype=float))
    neighbours = list(g.neighbors(cell_idx))
    nb_data = data_table[data_table['new_id'].isin(neighbours)]
    idx_data = data_table[data_table['new_id'] == cell_idx]
    cell_myo = np.array(idx_data[['frame_nb', 'concentration_myo']])
    cell_size = np.array(idx_data[['frame_nb', 'area_microns']])
    nb_myo_dict = {tf: np.mean(nb_data[nb_data['frame_nb'] == tf]['concentration_myo'])
                    for tf in idx_data['frame_nb'].unique()}
    nb_myo = np.array(list(nb_myo_dict.items()))
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax1.plot(cell_myo[:, 0], cell_myo[:, 1], color='blue')
    ax1.plot(nb_myo[:, 0], nb_myo[:, 1], color='red')
    ax2.plot(cell_size[:, 0], cell_size[:, 1])
    plt.show()


def get_samestage_cells(table, myo_conc, tolerance=0.2):
    cell2tf = {}
    for idx in table['new_id'].unique():
        idx_data = table[table['new_id'] == idx]
        closest_entry = idx_data.iloc[np.argmin(np.array(np.abs(idx_data['concentration_myo'] - myo_conc)))]
        tf, conc_myo = closest_entry['frame_nb'], closest_entry['concentration_myo']
        if np.abs(conc_myo - myo_conc) / myo_conc <= tolerance:
            cell2tf[idx] = tf
    cell2tf = np.array(list(cell2tf.items()))
    return cell2tf

def plot_samestage_neighb(table, myo, expand, constrict, n_pnts=3):
    g = graph.rag_boundary(segmentation[15], np.ones_like(segmentation[15], dtype=float))
    samemyo_cells = get_samestage_cells(table, myo_conc=myo, tolerance=0.05)
    for idx, tf in samemyo_cells:
        if not g.has_node(idx): continue
        if idx in constrict:
            color = 'blue'
        elif idx in expand:
            color = 'green'
        else:
            color = 'black'
        myo = get_avg_conc(idx, table, tf, n_pnts)
        myo_neighb = np.mean([get_avg_conc(i, table, tf, n_pnts) for i in g.neighbors(idx) if i!=0])
        plt.scatter(myo, myo_neighb, c=color)
    plt.show()

all_expand = [59, 71, 78, 77, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 96, 100, 2, 3, 4, 5, 6, 7, 9, 10, 89, 95, 103, 105, 106]
all_constrict = [44, 46, 48, 49, 51, 52, 53, 55, 56, 57, 58, 98, 101, 72, 74, 79, 81, 83, 85, 86, 87, 88, 102, 25, 27, 28, 29, 30, 31, 33, 34, 37, 40, 41, 42]
