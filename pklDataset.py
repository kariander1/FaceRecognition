import os

import torch
import pickle
from torch import Tensor
from typing import Tuple, Iterator
from contextlib import contextmanager
from torch.utils.data import Dataset, IterableDataset
import pandas as pd


class PklEmbeddingsDataset(Dataset):
    """
    A dataset representing embedded vectors of face recognition
    """

    def __init__(self, pkl_dir_path1, pkl_dir_path2):
        """
        :param pkl_dir_path1:
        :param pkl_dir_path2:
        """
        super().__init__()
        self.ds1 = PklDataset(pkl_dir_path1)
        self.ds2 = PklDataset(pkl_dir_path2)

        self.n_records = len(self.ds2)
        self.classes = self.ds2.classes

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image1, label1 = self.ds1[index]
        image2, label2 = self.ds2[index]
        assert label1 == label2
        return image1, image2
        # return next(self.iterator)

    def __len__(self):
        """
        :return: Number of samples in this dataset.
        """
        # ====== YOUR CODE: ======
        return self.n_records
        # ========================


class PklDataset(Dataset):
    """
    A dataset representing embedded vectors of face recognition
    """

    def __init__(self, pkl_dir_path, infer_classes_and_n_records=False):
        """
        :param pklFilePath:

        """
        super().__init__()
        print("Loading dataframe..")
        self.pkl_dir_path = pkl_dir_path
        with open(os.path.join(pkl_dir_path, 'metadata.pkl'), 'rb') as pkl_metadata:
            metadata_dict = pickle.load(pkl_metadata)
            self.n_records = metadata_dict["n_images"]
            self.classes = metadata_dict["classes"]
            self.dataset_batch_size = metadata_dict["batch_size"]
        self.files = [None] * (self.n_records // self.dataset_batch_size)
        if infer_classes_and_n_records:
            self._infer_classes_and_n_records()
        # self.iterator = self._get_image()

    def _infer_classes_and_n_records(self):
        labels_set = set()
        n_records = 0
        raw_data = self._load_batch(return_on_failure=True)
        while raw_data is not None:
            n_records += raw_data.shape[0]
            labels_set.update(raw_data[:, 0].cpu().detach().tolist())
            raw_data = self._load_batch(return_on_failure=True)
        self.n_records = n_records
        self.classes = [int(item) for item in labels_set]

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        i_batch = index // self.dataset_batch_size
        offset = index % self.dataset_batch_size
        batch_path = os.path.join(self.pkl_dir_path, str(i_batch) + '.pkl')
        # if self.files[i_batch] is None:
        # self.files[i_batch] = open(batch_path, 'rb')
        f = open(batch_path, 'rb')
        batch = pickle.load(f)
        label = int(batch[offset, 0].item())
        image = batch[offset, 1:]
        return image, label
        # return next(self.iterator)

    def _get_image(self) -> Tuple[Tensor, int]:
        while True:
            raw_data = self._load_batch()
            for feature_tensor in raw_data:
                label = int(feature_tensor[0].item())
                image = feature_tensor[1:]
                yield image, label

    def _load_batch(self, return_on_failure=False):

        try:
            raw_data = pickle.load(self.pkl_file)
        except EOFError:
            if return_on_failure:
                return None
            # print("Reopening pkl file...")
            self.pkl_file.close()
            self.pkl_file = open(self.pkl_file_path, 'rb')
            nRecords = pickle.load(self.pkl_file)
            classes = pickle.load(self.pkl_file)
            raw_data = pickle.load(self.pkl_file)
        return raw_data

    def __len__(self):
        """
        :return: Number of samples in this dataset.
        """
        # ====== YOUR CODE: ======
        return self.n_records
        # ========================

#
# class ImageStreamDataset(IterableDataset):
#     """
#     A dataset representing an infinite stream of noise images of specified dimensions.
#     """
#
#     def __init__(self, num_classes: int, C: int, W: int, H: int):
#         """
#         :param num_classes: Number of classes (labels)
#         :param C: Number of channels per image
#         :param W: Image width
#         :param H: Image height
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.image_dim = (C, W, H)
#
#     def __iter__(self) -> Iterator[Tuple[Tensor, int]]:
#         """
#         :return: An iterator providing an infinite stream of random labelled images.
#         """
#
#         # TODO:
#         #  Yield tuples to produce an iterator over random images and labels.
#         #  The iterator should produce an infinite stream of data.
#         # ====== YOUR CODE: ======
#         while True:
#             # Generate infinite images as requested and yield each time
#             yield random_labelled_image(self.image_dim, self.num_classes)
#         # ========================
#
