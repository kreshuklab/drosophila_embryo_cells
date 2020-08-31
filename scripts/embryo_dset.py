from torch.utils.data.dataset import Dataset
from scipy.ndimage import find_objects


class EmbryoDataset(Dataset):
    def __init__(self, indices, segm_data, transforms,
                 membrane_data=None, myosin_data=None):
        if membrane_data: assert membrane_data.shape == segm_data.shape
        if myosin_data: assert myosin_data.shape == segm_data.shape
        self.indices = indices
        self.segm_data = segm_data
        self.membrane_data = membrane_data
        self.myosin_data = myosin_data
        self.bbs = self.get_bbs(segm_data)
        self.transforms = transforms

    def get_bbs(self, segm_image):
        bbs = find_objects(segm_image)
        # find_objects ignores label 0, but it's handy to index like this
        bbs.insert(0, None)
        return bbs

    def __len__(self):
        return len(self.indices)


class ClassEmbryoDataset(EmbryoDataset):
    def __init__(self, class_labels, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(class_labels, dict)
        self.class_labels = class_labels
        self.max_label = max(list(class_labels.values()))
        self.min_label = min(list(class_labels.values()))

    # we train as regression, not classification, so it makes sense to normalize the classes
    def norm_label(self, label):
        return (label - self.min_label) / (self.max_label - self.min_label)

    def __getitem__(self, idx):
        cell_id = self.indices[idx]
        cell_class = self.class_labels[cell_id]
        cell_mask = self.segm_data[self.bbs[cell_id]] == cell_id
        return self.transforms(cell_mask), self.norm_label(cell_class)
