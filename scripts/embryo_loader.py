import h5py
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from inferno.io.transform import Compose
from inferno.io.transform import generic as gen_transf
from inferno.io.transform import image as img_transf
from inferno.utils.io_utils import yaml2dict
from embryo_dset import ClassEmbryoDataset


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
    if transform_config.get('cast'):
        transforms.add(gen_transf.Cast('float32'))
    if transform_config.get('normalize_range'):
        normalize_range_config = transform_config.get('normalize_range')
        transforms.add(gen_transf.NormalizeRange(**normalize_range_config))
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
        data_path = self.config.get('data_path')
        table_path = self.config.get('table_path')
        with h5py.File(data_path, 'r') as f:
            self.membranes = f['membranes'][:]
            self.myosin = f['myosin'][:]
            self.segmentation = f['segmentation'][:]
        data_table = pd.read_csv(table_path)
        self.tf2ids = self.get_tf_dict(data_table)
        self.id2row = self.get_row_dict(data_table)
        self.tf = self.config.get('timeframe', None)
        self.transforms = [get_transforms(self.config.get(key))
                           for key in ['train_transforms', 'val_transforms']]

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

    def get_class_loaders(self):
        segm_frame = self.segmentation[self.tf]
        split_labels = get_train_val_split(self.tf2ids[self.tf])
        cell_dsets = [ClassEmbryoDataset(class_labels=self.id2row, indices=lbls,
                                         segm_data=segm_frame, transforms=tfs,
                                         **self.config.get('dataset_kwargs', {}))
                      for lbls, tfs in zip(split_labels, self.transforms)]
        train_loader = DataLoader(cell_dsets[0], **self.config.get('loader_config'))
        val_loader = DataLoader(cell_dsets[1], **self.config.get('val_loader_config'))
        return train_loader, val_loader
