import PIL
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import os


class CelebADataset(Dataset):

    def __init__(self, directory, transform=None, split="train"):
        self.transform = transform
        self.directory = directory
        self.split = split

        self.filenames = self.load_filenames()
        self.img_dir = "img_align_celeba"

    def load_filenames(self):

        partition_idx = {
            0: 'train',
            1: 'val',
            2: 'test'
        }

        partitions_file = os.path.join(self.directory, 'list_eval_partition.txt')
        partitions = pd.read_csv(partitions_file, sep=' ', names=['filename', 'partition'])
        partitions.partition = partitions.partition.map(lambda x: partition_idx[x])
        print(partitions.partition.unique())
        return partitions[partitions.partition == self.split].filename.tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        img_path = os.path.join(self.directory, self.img_dir, filename)

        img: Image = PIL.Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # ToDo: parse a target
        target = None

        return img


if __name__ == '__main__':

    import torch
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    celeba = CelebADataset("/Users/tim/Downloads/celeba",
                           split="test", transform=transform)

    print(celeba[42][0].shape)

    plt.imshow((celeba[42][0].transpose(2, 0).transpose(1, 0) * 0.5 + 0.5) )
    plt.show()