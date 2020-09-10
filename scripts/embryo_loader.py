import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from inferno.io.transform import Compose
from inferno.io.transform import generic as gen_transf
from inferno.io.transform import image as img_transf
from inferno.utils.io_utils import yaml2dict
from embryo_dset import *


def get_train_val_split(labels, split=0.2, r_seed=None):
    np.random.seed(seed=r_seed)
    np.random.shuffle(labels)
    spl = int(np.floor(len(labels)*split))
    return labels[spl:], labels[:spl]


def get_transforms(transform_config):
    transforms = Compose()
    if transform_config.get('crop_pad_to_size'):
        crop_pad_to_size = transform_config.get('crop_pad_to_size')
        transforms.add(img_transf.CropPad2Size(**crop_pad_to_size))
    if transform_config.get('normalize'):
        normalize_config = transform_config.get('normalize')
        transforms.add(gen_transf.Normalize(**normalize_config))
    if transform_config.get('normalize_range'):
        normalize_range_config = transform_config.get('normalize_range')
        transforms.add(gen_transf.NormalizeRange(**normalize_range_config))
    if transform_config.get('cast'):
        transforms.add(gen_transf.Cast('float32'))
    if transform_config.get('flip'):
        transforms.add(img_transf.RandomFlip())
    if transform_config.get('rotate'):
        transforms.add(img_transf.RandomRotate())
    if transform_config.get('transpose'):
        transforms.add(img_transf.RandomTranspose())
    if transform_config.get('elastic_transform'):
        elastic_config = transform_config.get('elastic_transform')
        transforms.add(img_transf.ElasticTransform(order=3, **elastic_config))
    if transform_config.get('noise'):
        noise_config = transform_config.get('noise')
        transforms.add(img_transf.AdditiveGaussianNoise(**noise_config))
    if transform_config.get('torch_batch'):
        transforms.add(gen_transf.AsTorchBatch(2))
    return transforms


class EmbryoDataloader():
    def __init__(self, configuration_file):
        self.config = yaml2dict(configuration_file)
        self.data_path = self.config.get('data_path')
        table_path = self.config.get('table_path')
        data_table = pd.read_csv(table_path)
        self.id2row = self.get_row_dict(data_table)
        self.tf = self.config.get('timeframe', slice(None, None))
        self.transforms = [get_transforms(self.config.get(key))
                           for key in ['train_transforms', 'val_transforms']]
        self.basic_trfs = get_transforms(self.config.get('basic_transforms'))
        mode = self.config.get('training_type', 'segm_class')
        if mode == 'segm_class':
            self.dset = ClassSegmDataset
        elif mode == 'myo_class':
            self.dset = ClassMyoDataset
        elif mode == 'align_segm_class':
            self.dset = ClassAlignSegmDataset
        elif mode == 'align_myo_class':
            self.dset = ClassAlignMyoDataset

        if mode.startswith('align'):
            myo_c = self.config.get('myo_concentration')
            thres = self.config.get('reject_threshold', 0.2)
            self.labels = self.get_samestage_cells(data_table, myo_conc=myo_c, tolerance=thres)
        else:
            tf2ids = self.get_tf_dict(data_table)
            self.labels = tf2ids[self.tf]

    def get_tf_dict(self, table):
        tf_dict = {}
        for tf in table['frame_nb'].unique():
            ids_present = table[table['frame_nb'] == tf]['new_id'].unique()
            tf_dict[tf] = ids_present
        return tf_dict

    def get_row_dict(self, table):
        row_dict = {}
        for i in table['new_id'].unique():
            row = table[table['new_id'] == i]['row_id'].unique()
            assert len(row) == 1, "Id {} is assigned to multiple rows: {}".format(i, row)
            row_dict[i] = row[0]
        return row_dict

    def get_samestage_cells(self, table, myo_conc, tolerance=0.2):
        cell2tf = {}
        for idx in table['new_id'].unique():
            idx_data = table[table['new_id'] == idx]
            closest_entry = idx_data.iloc[np.argmin(np.array(np.abs(idx_data['concentration_myo'] - myo_conc)))]
            tf, conc_myo = closest_entry['frame_nb'], closest_entry['concentration_myo']
            if np.abs(conc_myo - myo_conc) / myo_conc <= tolerance:
                cell2tf[idx] = tf
        cell2tf = np.array(list(cell2tf.items()))
        return cell2tf

    def get_class_loaders(self):
        split_labels = get_train_val_split(self.labels)
        cell_dsets = [self.dset(class_labels=self.id2row, indices=lbls, timeframe=self.tf,
                                data_file=self.data_path, transforms=trfs,
                                **self.config.get('dataset_kwargs', {}))
                      for lbls, trfs in zip(split_labels, self.transforms)]
        train_loader = DataLoader(cell_dsets[0], **self.config.get('loader_config'))
        val_loader = DataLoader(cell_dsets[1], **self.config.get('val_loader_config'))
        return train_loader, val_loader

    def get_predict_loader(self):
        lbls = self.labels
        trfs = self.basic_trfs
        cell_dset = self.dset(class_labels=self.id2row, indices=lbls, timeframe=self.tf,
                              data_file=self.data_path, transforms=trfs,
                              **self.config.get('dataset_kwargs', {}))
        predict_loader = DataLoader(cell_dset, **self.config.get('val_loader_config'))
        return predict_loader
