import os
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

class SegmentationDataset(data.Dataset):
    def __init__(self, dataset_path) -> None:
        input_path = os.path.join(dataset_path,"input")
        output_path = os.path.join(dataset_path,"label")

        for root,folders, files in os.walk(input_path):
            self.input_file_list = [os.path.join(root,file) for file in files]

        for root,folders, files in os.walk(output_path):
            self.segment_file_list = [os.path.join(root,file) for file in files]

        print(f"Read {len(self.input_file_list)} valid input samples, and {len(self.segment_file_list)} annotation")

    def __getitem__(self, index):
        input_image = torch.from_numpy(np.load(self.input_file_list[index], allow_pickle=True))

        label = torch.from_numpy(np.load(self.segment_file_list[index], allow_pickle=True))

        return input_image, label

    def get_file_path(self, index):
        return self.input_file_list[index]

    def __len__(self):
        return len(self.input_file_list)