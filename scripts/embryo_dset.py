import h5py
import numpy as np
from torch.utils.data.dataset import Dataset
from scipy.ndimage import find_objects


ALL_ROWS = [3, 4, 5, 6, 7, 8]


class EmbryoDataset(Dataset):
    def __init__(self, indices, data_file, timeframe, transforms):
        with h5py.File(data_file, 'r') as f:
            self.membrane_data = f['membranes'][timeframe]
            self.myosin_data = f['myosin'][timeframe]
            self.segm_data = f['segmentation'][timeframe]
        self.indices = indices
        self.bbs = self.get_bbs(self.segm_data)
        self.transforms = transforms

    def get_bbs(self, segm_image):
        bbs = find_objects(segm_image)
        # find_objects ignores label 0, but it's handy to index like this
        bbs.insert(0, None)
        return bbs

    def __len__(self):
        return len(self.indices)


class ClassSegmDataset(EmbryoDataset):
    def __init__(self, class_labels, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(class_labels, dict)
        self.class_labels = class_labels

    # we train as regression, not classification, so it makes sense to normalize the classes
    def norm_label(self, label):
        return (label - min(ALL_ROWS)) / (max(ALL_ROWS) - min(ALL_ROWS))

    def __getitem__(self, idx):
        cell_id = self.indices[idx]
        cell_class = self.class_labels[cell_id]
        cell_mask = self.segm_data[self.bbs[cell_id]] == cell_id
        return self.transforms(cell_mask), self.norm_label(cell_class)


class ClassMyoDataset(ClassSegmDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        cell_id = self.indices[idx]
        cell_class = self.class_labels[cell_id]
        cell_mask = self.segm_data[self.bbs[cell_id]] == cell_id
        myo_mask = self.myosin_data[self.bbs[cell_id]] * cell_mask
        return self.transforms(myo_mask), self.norm_label(cell_class)


class ClassAlignSegmDataset(ClassSegmDataset):
    def __init__(self, indices, *args, **kwargs):
        # we expect [idx, tf]
        assert indices.shape[-1] == 2
        self.idx2tf = {idx: tf for idx, tf in indices}
        super().__init__(*args, indices=indices[:, 0], **kwargs)
        rows_present = set([self.class_labels[i] for i in indices[:, 0]])
        # we need to make sure we have all classes present
        print("The number of samples per class is /n",
              np.unique([self.class_labels[i] for i in indices[:, 0]], return_counts=True))
        assert all(i in rows_present for i in ALL_ROWS)

    def get_bbs(self, segm_volume):
        idx2bb = {idx : find_objects(segm_volume[row] == idx)[0]
                  for idx, row in self.idx2tf.items()}
        return idx2bb

    def __getitem__(self, idx):
        cell_id = self.indices[idx]
        cell_class = self.class_labels[cell_id]
        cell_mask = self.segm_data[self.idx2tf[cell_id]][self.bbs[cell_id]] == cell_id
        return self.transforms(cell_mask), self.norm_label(cell_class)


class ClassAlignMyoDataset(ClassAlignSegmDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        cell_id = self.indices[idx]
        cell_class = self.class_labels[cell_id]
        cell_mask = self.segm_data[self.idx2tf[cell_id]][self.bbs[cell_id]] == cell_id
        myo_mask = self.myosin_data[self.idx2tf[cell_id]][self.bbs[cell_id]] * cell_mask
        return self.transforms(myo_mask), self.norm_label(cell_class)
