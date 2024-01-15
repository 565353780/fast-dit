import os
import torch
import numpy as np
from torch.utils.data import Dataset


class ASDFDataset(Dataset):
    def __init__(self, asdf_dataset_folder_path: str) -> None:
        self.asdf_dataset_folder_path = asdf_dataset_folder_path

        self.asdf_file_list = list(range(1000))
        self.mesh_file_list = list(range(1000))
        return

    def __len__(self):
        assert len(self.asdf_file_list) == len(
            self.mesh_file_list
        ), "Number of feature files and label files should be same"
        return len(self.asdf_file_list)

    def __getitem__(self, idx):
        asdf_file_path = self.asdf_file_list[idx]
        mesh_file_path = self.mesh_file_list[idx]

        asdf = np.random.rand(1, 100, 40)
        context = np.random.rand(1, 100, 40)

        return torch.from_numpy(asdf).type(torch.float32), torch.from_numpy(context).type(torch.float32)
