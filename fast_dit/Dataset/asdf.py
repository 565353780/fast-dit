import os
import torch
import numpy as np
from tqdm import tqdm
from random import choice
from torch.utils.data import Dataset

from fast_dit.Config.fp import USE_FP_16


class ASDFDataset(Dataset):
    def __init__(self, asdf_dataset_folder_path: str) -> None:
        self.asdf_file_list = []
        self.context_files_list = []

        self.loadDataset(asdf_dataset_folder_path)
        return

    def loadDataset(self, asdf_dataset_folder_path: str) -> bool:
        class_foldername_list = os.listdir(asdf_dataset_folder_path)

        for class_foldername in class_foldername_list:
            model_folder_path = asdf_dataset_folder_path + class_foldername + '/'
            if not os.path.exists(model_folder_path):
                continue

            model_filename_list = os.listdir(model_folder_path)

            for model_filename in tqdm(model_filename_list):
                asdf_folder_path = model_folder_path + model_filename + '/'
                if not os.path.exists(asdf_folder_path):
                    continue

                asdf_filename_list = os.listdir(asdf_folder_path)

                if 'final.npy' not in asdf_filename_list:
                    continue

                context_files = []

                for asdf_filename in asdf_filename_list:
                    if asdf_filename == 'final.npy':
                        continue

                    if asdf_filename[-4:] != '.npy':
                        continue

                    context_files.append(asdf_folder_path + asdf_filename)

                self.asdf_file_list.append(asdf_folder_path + 'final.npy')
                self.context_files_list.append(context_files)

        return True

    def __len__(self):
        assert len(self.asdf_file_list) == len(
            self.context_files_list
        ), "Number of feature files and label files should be same"
        return len(self.asdf_file_list)

    def __getitem__(self, idx):
        asdf_file_path = self.asdf_file_list[idx]
        context_file_path = choice(self.context_files_list[idx])

        asdf = np.load(asdf_file_path, allow_pickle=True).item()['params'].reshape(1, 100, 40)
        context = np.load(context_file_path, allow_pickle=True).item()['params'].reshape(1, 100, 40)

        if USE_FP_16:
            return torch.from_numpy(asdf).type(torch.float16), torch.from_numpy(context).type(torch.float16)

        return torch.from_numpy(asdf).type(torch.float32), torch.from_numpy(context).type(torch.float32)
