import argparse
import os
import h5py
import numpy as np
import pandas as pd
from tifffile import imread


def get_segm(tif_folder):
    num_frames = len(os.listdir(tif_folder))
    segm_timepoints = []
    for i in range(num_frames):
        tif_file = os.path.join(tif_folder, 'cell_identity_T{}.tif'.format(i))
        segm_t = imread(tif_file)[:, :, 2]
        segm_timepoints.append(segm_t)
    segm_timepoints = np.array(segm_timepoints)
    return segm_timepoints


def remap_segm(segm_timepoints, data_frame):
    remapped_segm = np.zeros_like(segm_timepoints)
    track_id_map = {tr_id: new_id + 1 for new_id, tr_id
                    in enumerate(data_frame['track_id_cells'].unique())}
    for row in data_frame.iterrows():
        tf, track_id, loc_id = row[1][['frame_nb', 'track_id_cells', 'local_id_cells']]
        new_id = track_id_map[track_id]
        remapped_segm[tf][segm_timepoints[tf] == loc_id] = new_id
    return remapped_segm, track_id_map


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert tiff data to h5')
    parser.add_argument('raw_tif_file', type=str,
                        help='tif file with raw data')
    parser.add_argument('tif_folder', type=str,
                        help='path to segmentation tifs')
    parser.add_argument('tracking_csv', type=str,
                        help='table with tracking output')
    parser.add_argument('out_h5', type=str,
                        help='output file')
    args = parser.parse_args()

    raw_data = imread(args.raw_tif_file)[2:33]
    segmentation = get_segm(args.tif_folder)

    data_table = pd.read_csv(args.tracking_csv)

    remapped_segmentation, id_map = remap_segm(segmentation, data_table)
    data_table['new_id'] = [id_map[key] for key in data_table['track_id_cells']]
    data_table.to_csv(args.tracking_csv, index=False)

    with h5py.File(args.out_h5, 'w') as f:
        _ = f.create_dataset('membranes', data=raw_data[:, 0], compression='gzip')
        _ = f.create_dataset('myosin', data=raw_data[:, 1], compression='gzip')
        _ = f.create_dataset('segmentation', data=remapped_segmentation, compression='gzip')
