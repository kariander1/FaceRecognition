import torch
import pickle
from torch import Tensor
from typing import Tuple, Iterator
from contextlib import contextmanager
from torch.utils.data import Dataset, IterableDataset
import pandas as pd

class PklDataset(Dataset):
    """
    A dataset representing embedded vectors of face recognition
    """

    def __init__(self, pklFilePath):
        """
        :param pklFilePath:

        """
        super().__init__()
        print("Loading dataframe..")
        self.dataframe = pd.read_pickle(pklFilePath)
        self.nRecords = self.dataframe.shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Returns a labeled sample.
        :param index: Sample index.
        :return: A tuple (sample, label) containing the image and its class label.
        Raises a ValueError if index is out of range.
        """

        # TODO:
        #  Create a random image tensor and return it.
        #  Make sure to always return the same image for the
        #  same index (make it deterministic per index), but don't mess-up
        #  the random state outside this method.
        #  Raise a ValueError if the index is out of range.
        # ====== YOUR CODE: ======
        raw_data = self.dataframe.iloc[index]

        label = raw_data.iloc[0]
        image = torch.tensor(raw_data.iloc[1:].values)
        # if index >= self.num_samples or index < 0:
        #     raise ValueError()
        # with torch_temporary_seed(index):
        #     (image, label) = random_labelled_image(self.image_dim, self.num_classes)
        return image, label
        # ========================

    def __len__(self):
        """
        :return: Number of samples in this dataset.
        """
        # ====== YOUR CODE: ======
        return self.nRecords
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
