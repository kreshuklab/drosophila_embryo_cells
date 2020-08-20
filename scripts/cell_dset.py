from torch.utils.data.dataset import Dataset
from scipy.ndimage import find_objects

class EmbryoDataset(Dataset):
    def __init__(self, indices, segm_data, membrane_data=None, myosin_data=None):
        if membrane_data: assert membrane_data.shape == segm_data.shape
        if myosin_data: assert myosin_data.shape == segm_data.shape
        self.indices = indices
        self.segm_data = segm_data
        self.membrane_data = membrane_data
        self.myosin_data = myosin_data
        self.bbs = self.get_bbs(segm_data)

    def get_bbs(self, segm_image):
        bbs = find_objects(segm_image)
        bbs.insert(0, None)
        return bbs

    def __len__(self):
        return len(self.indices)


class ClassEmbryoDataset(EmbryoDataset):
    def __init__(self, indices, segm_data, class_labels):
        super().__init__(indices, segm_data)
        assert isinstance(class_labels, dict)
        self.class_labels = class_labels

    def __getitem__(self, idx):
        cell_id = self.indices[idx]
        print(cell_id)
        cell_class = self.class_labels[cell_id]
        cell_mask = self.segm_data[self.bbs[cell_id]] == cell_id
        return cell_mask, cell_class
