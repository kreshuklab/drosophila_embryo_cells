import argparse
from itertools import chain
import os
import h5py
import numpy as np
import pandas as pd
from tifffile import imread
from skimage.measure import label


def get_segm(tif_folder, big_thres=10000):
    num_frames = len(os.listdir(tif_folder))
    segm_timepoints = []
    for i in range(num_frames):
        tif_file = os.path.join(tif_folder, 'cell_identity_T{}.tif'.format(i))
        segm_t = imread(tif_file)[:, :, 2]
        # because one shouldn't save labels in rgb, that's why
        segm_t = label(segm_t, background=0)
        segm_t = remove_big_segments(segm_t, big_thres)
        segm_t = remove_border_cells(segm_t)
        segm_timepoints.append(segm_t)
    segm_timepoints = np.array(segm_timepoints)
    return segm_timepoints


def remove_big_segments(image, trsh):
    labels, counts = np.unique(image, return_counts=True)
    for lbl in labels[counts > trsh]:
        image[image == lbl] = 0
    return image


def remove_border_cells(image):
    image_borders = image.copy()
    image_borders[2:-2, 2:-2] = 0
    border_cells = np.unique(image_borders)
    border_cells = border_cells[border_cells != 0]
    for cell in border_cells:
        image[image == cell] = 0
    return image


def remap_segm(segm_timepoints, data_frame):
    remapped_segm = np.zeros_like(segm_timepoints)
    track_id_map = {tr_id: new_id + 1 for new_id, tr_id
                    in enumerate(data_frame['track_id_cells'].unique())}
    for row in data_frame.iterrows():
        tf, track_id, loc_id = row[1][['frame_nb', 'track_id_cells', 'local_id_cells']]
        new_id = track_id_map[track_id]
        remapped_segm[tf][segm_timepoints[tf] == loc_id] = new_id
    return remapped_segm, track_id_map


def get_track(segm, idx, start_tf):
    track = ['{}_{}'.format(start_tf, idx)]
    for i in range(start_tf + 1, len(segm)):
        cell_mask = segm[i - 1] == idx
        new_ids, counts = np.unique(segm[i][cell_mask], return_counts=True)
        idx = new_ids[np.argmax(counts)]
        if np.max(counts) < np.sum(cell_mask) * 0.5:
            idx = 0
        if idx == 0:
            break
        track.append('{}_{}'.format(i, idx))
    return track


def visualize_track(track, segm):
    segm_track = np.zeros_like(segm)
    for item in track:
        tf, idx = item.split('_')
        segm_track[int(tf)] = segm[int(tf)] == int(idx)
    viewer.add_labels(segm_track, blending='additive')


def get_track_list(segm):
    track_list = []
    for tf, segm_tf in enumerate(segm[:-1]):
        track_set = set(chain.from_iterable(track_list))
        this_frame_ids = np.unique(segm_tf)
        for idx in this_frame_ids:
            if '{}_{}'.format(tf, idx) not in track_set:
                idx_track = get_track(segm, idx, tf)
                if len(idx_track) < 4:
                    continue
                track_list.append(idx_track)
    return track_list


def remap_segmentation(segm):
    tracks = get_track_list(segm)
    new_segm = np.zeros_like(segm)
    for new_id, track in enumerate(tracks):
        for item in track:
            tf, idx = item.split('_')
            new_segm[int(tf)][segm[int(tf)] == int(idx)] = new_id + 1
    return new_segm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert tiff data to h5')
    parser.add_argument('data_path', type=str,
                        help='folder with the data')
    parser.add_argument('sample_num', type=int, choices=[5, 6, 24],
                        help='the sample number')
    args = parser.parse_args()

    raw_tif_file = os.path.join(args.data_path, 'Img{}_SUM.tif'.format(args.sample_num))
    segm_folder = os.path.join(args.data_path, 'Img{}_segmentation/'.format(args.sample_num))
    out_h5 = os.path.join(args.data_path, 'img{}_new.h5'.format(args.sample_num))

    raw_data = imread(raw_tif_file)
    segmentation = get_segm(segm_folder)

    remapped_segmentation = remap_segmentation(segmentation)

    if args.sample_num == 5:
        raw_data = raw_data[2:33]
    elif args.sample_num == 6:
        raw_data = raw_data[:23]
    elif args.sample_num == 24:
        raw_data = raw_data[:22]
        remapped_segmentation = remapped_segmentation[:22]

    with h5py.File(out_h5, 'w') as f:
        _ = f.create_dataset('membranes', data=raw_data[:, 0], compression='gzip')
        _ = f.create_dataset('myosin', data=raw_data[:, 1], compression='gzip')
        _ = f.create_dataset('segmentation', data=remapped_segmentation, compression='gzip')
