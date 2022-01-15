import math
import os
import random

import copy
import torch
import pickle
from torch import Tensor
from typing import Tuple, Iterator
from contextlib import contextmanager
from torch.utils.data import Dataset, IterableDataset
import pandas as pd


def SplitDataset(full_dataset, n_labels,val_ratio=0.1, test_ratio=0.1):
    dataset_size = len(full_dataset)
    label_train = n_labels*(1-val_ratio-test_ratio)
    label_val = n_labels * (1 - val_ratio)
    label_test = n_labels
    batch_size = full_dataset.ds1.dataset_batch_size

    train_dataset =copy.deepcopy(full_dataset)
    train_dataset.n_records = 14031*batch_size

    val_dataset = copy.deepcopy(full_dataset)
    val_dataset.n_records = (15789-14032) * batch_size
    val_dataset.ds1.batch_offset = 14032
    val_dataset.ds2.batch_offset = 14032

    test_dataset = copy.deepcopy(full_dataset)
    test_dataset.n_records = (17481 - 15790) * batch_size
    test_dataset.ds1.batch_offset = 15780
    test_dataset.ds2.batch_offset = 15780

    return train_dataset,val_dataset,test_dataset

    # n_batches = dataset_size//full_dataset.ds1.dataset_batch_size
    # i = 15790*full_dataset.ds1.dataset_batch_size
    # for i_batch in range(15790,n_batches):
    #     _,_,label = full_dataset[i]
    #     if label>=label_test:
    #         break
    #     i += full_dataset.ds1.dataset_batch_size

#
#
# def SplitDataset(full_dataset, val_ratio=0.1, test_ratio=0.1):
#     dataset_size = len(full_dataset)
#     indices = list(range(dataset_size))
#
#     random.shuffle(indices)
#     val_percent = int(math.floor(val_ratio * dataset_size))
#     test_percent = int(math.floor(test_ratio * dataset_size))
#     train_indices, val_indices, test_indices = indices[val_percent + test_percent:], \
#                                                indices[test_percent:val_percent + test_percent], \
#                                                indices[:test_percent]
#     new_lengths = [dataset_size - (val_percent + test_percent), val_percent, test_percent]
#     new_datasets = [full_dataset[train_indices], full_dataset[val_indices], full_dataset[test_indices]]
#     for i_dataset, dataset in enumerate(new_datasets):
#         dataset.n_records = new_lengths[i_dataset]
#     return


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
        image1, label1, idx1 = self.ds1[index]
        image2, label2, idx2 = self.ds2[index]
        assert idx1 == idx2
        assert label1 == label2
        return image1, image2, label1
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
        print("Loading pkl dataset: "+pkl_dir_path )
        self.pkl_dir_path = pkl_dir_path
        with open(os.path.join(pkl_dir_path, 'metadata.pkl'), 'rb') as pkl_metadata:
            metadata_dict = pickle.load(pkl_metadata)
            self.n_records = metadata_dict["n_images"]
            self.classes = metadata_dict["classes"]
            self.dataset_batch_size = metadata_dict["batch_size"]
        self.files = [None] * (self.n_records // self.dataset_batch_size)
        self.batch_offset = 0;
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

    def __getitem__(self, index: int) -> Tuple[Tensor, int,int]:
        i_batch = self.batch_offset + index // self.dataset_batch_size
        offset = index % self.dataset_batch_size
        batch_path = os.path.join(self.pkl_dir_path, str(i_batch) + '.pkl')
        # if self.files[i_batch] is None:
        # self.files[i_batch] = open(batch_path, 'rb')
        f = open(batch_path, 'rb')
        batch = pickle.load(f)
        idx = int(batch[offset, 0].item())
        label = int(batch[offset, 1].item())
        image = batch[offset, 2:]
        return image, label,idx
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
