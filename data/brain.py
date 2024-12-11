from matplotlib import pyplot as plt
import nibabel as nib
import os
import torch
from torch.utils.data import Dataset


class BrainDataset(Dataset):

    def __init__(self, path,
                 extension='nii.gz',
                 scan_type='flair',
                 min_slice_depth=80,
                 max_slice_depth=115,
                 depth_stride=5):
        self.path = path
        self.tumors = []
        self.slice_depths = range(min_slice_depth, max_slice_depth, depth_stride)
        self.num_slices = len(self.slice_depths)
        self.scan_type = scan_type
        self.extension = extension

        for tumor in os.listdir(path):
            if not os.path.isdir(os.path.join(path, tumor)):
                continue
            else:
                self.tumors.append(tumor)

    def __len__(self):
        return len(self.tumors) * self.num_slices

    def __getitem__(self, idx):
        tumor = self.tumors[idx // self.num_slices]
        sidx = idx % self.num_slices
        slice_depth = self.slice_depths[sidx]
        timg_path = os.path.join(self.path, tumor, f'{tumor}_{ self.scan_type }.{ self.extension }')
        timg = nib.load(timg_path)
        timg = torch.as_tensor(timg.get_fdata()[:, :, slice_depth] / 3000.0, dtype=torch.float32)
        timg = torch.clamp(timg, 0., 1.)
        return torch.unsqueeze(timg, dim=0)


if __name__ == '__main__':

    dataset = BrainDataset(path="/Users/tim/Documents/GTD/04 - Tumor Growth/BraTS/data")
    for timg in dataset:
        plt.imshow(torch.squeeze(timg).numpy(), cmap='Greys_r', vmin=0., vmax=1.)
        plt.show()