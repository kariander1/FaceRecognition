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

def GetStartingIndices(full_dataset):
    os.makedirs('./starting_indices', exist_ok=True)
    pkl_file = './starting_indices/index.pkl'
    if os.path.isfile(pkl_file):
        with open(pkl_file, 'rb') as fid:
            index = pickle.load(fid)
        return index

    dl = torch.utils.data.DataLoader(full_dataset, 1, shuffle=False, drop_last=False)
    last_y = -1
    indices = []
    for i, batch in enumerate(dl):
        _, _, y = batch
        y= y.item()
        if y != last_y:
            print("Added index for label "+str(y))
            indices += [i]
        last_y = y

    index = torch.tensor(indices)
    with open(pkl_file, 'wb') as file:
        pickle.dump(index , file)

    return index


def SplitDataset(full_dataset, n_labels, ratios=[]):
    index = GetStartingIndices(full_dataset)
    sets = []
    new_ratios = [0]+ratios
    index_offset = 0
    current_ratio = 1 - sum(new_ratios)
    for i in range(0, len(new_ratios)):
        current_ratio += new_ratios[i]
        last_label = int(n_labels * current_ratio - 1)
        last_image_index = index[last_label+1].item()-1

        set = copy.deepcopy(full_dataset)
        set.n_records = last_image_index-index_offset+1
        set.ds1.index_offset = index_offset
        set.ds2.index_offset = index_offset
        sets += [set]
        index_offset = last_image_index+1
    return sets



def CreateNNLabels(dataset, dataset_name,force_run = False):
    os.makedirs('./nn_spaces', exist_ok=True)
    pkl_nn_space_file = './nn_spaces/nn_space_' + dataset_name + '.pkl'
    if os.path.isfile(pkl_nn_space_file) and not force_run:
        with open(pkl_nn_space_file, 'rb') as fid:
            nn_space_dict = pickle.load(fid)
        return nn_space_dict

    print("Creating NN full space for: ", dataset_name)

    dl = torch.utils.data.DataLoader(dataset, 1, shuffle=False, drop_last=False)
    embedding_space = []
    labels = []
    for i, batch in enumerate(dl):
        _, X2, y = batch
        y = y.item()

        embedding_space += [X2]
        labels += [y]

    print("Dumping embedding space")
    embedding_dict = {'features': torch.stack(embedding_space).squeeze(),
                      'labels': torch.tensor(labels)}
    with open(pkl_nn_space_file, 'wb') as file:
        pickle.dump(embedding_dict, file)
    return embedding_dict


def CreateMeanLabels(dataset, dataset_name, force_create=False):
    os.makedirs('./nn_spaces', exist_ok=True)
    pkl_nn_space_file = './nn_spaces/nn_space_' + dataset_name + '.pkl'
    if os.path.isfile(pkl_nn_space_file) and not force_create:
        with open(pkl_nn_space_file, 'rb') as fid:
            nn_space_dict = pickle.load(fid)
        return nn_space_dict

    print("Creating NN space for: ",dataset_name)
    index = GetStartingIndices(dataset)
    dl = torch.utils.data.DataLoader(dataset, 1, shuffle=False, drop_last=False)
    current_label_features = []
    embedding_space = []
    labels = []
    last_y = None

    for i, batch in enumerate(dl):
        _, X2, y = batch
        y = y.item()
        if last_y is None:
            last_y = y
        if y != last_y:
            print("Processed Label: ", last_y)
            mean_feature = torch.stack(current_label_features).squeeze()
            if len(mean_feature.shape) > 1:
                mean_feature = mean_feature.mean(dim=0)

            embedding_space += [mean_feature]
            labels += [last_y]
            current_label_features = []
            last_y = y

        current_label_features += [X2]

    print("Processed Label: ", last_y)
    mean_feature = torch.stack(current_label_features).squeeze()
    if len(mean_feature.shape) > 1:
        mean_feature = mean_feature.mean(dim=0)

    embedding_space += [mean_feature]
    labels += [last_y]

    print("Dumping embedding space")
    embedding_dict = {'features': torch.stack(embedding_space).squeeze(),
                      'labels': torch.tensor(labels)}
    with open(pkl_nn_space_file, 'wb') as file:
        pickle.dump(embedding_dict , file)
    return embedding_dict


def GetLabelVectors(label, index, lengths, full_dataset):
    i_label_start = index[label].item()
    i_label_stop = i_label_start + lengths[label].item()
    x1_vectors, x2_vectors = [], []
    for i in range(i_label_start, i_label_stop):
        x1, x2, y = full_dataset[i]
        assert y == label
        x1_vectors += [x1]
        x2_vectors += [x2]

    x1 = torch.stack(x1_vectors)
    x2 = torch.stack(x2_vectors)
    return x1,x2


def InterpolateDatasetRandom(dataset, full_dataset, min_interp_num=20, max_interp_num=30,interpolated_div = 2):
    set = copy.deepcopy(dataset)
    set_interpolated_data = copy.deepcopy(dataset)
    index = GetStartingIndices(full_dataset)
    _,_,label_first = set[0]
    _, _, label_last = set[len(set)-1]
    index = index[label_first:label_last+1]
    indices_shift = torch.hstack([index[1:] , torch.tensor(len(set))])
    lengths = indices_shift - index
    [sorted, idx_sorted] = torch.sort(lengths, descending=True)

    current_label = max(set.classes) + 1

    pkl_batch_size =  set.ds1.dataset_batch_size
    n_images = len(full_dataset)
    pkl_file_i = n_images // pkl_batch_size
    max_idx_offset = len(full_dataset) - len(set)
    max_idx = len(set) - 1
    interpolated_cnt = 959872
    if interpolated_cnt==0:
        paths = ['./Insight/insightface/recognition/arcface_torch/r18_features',
                 './Insight/insightface/recognition/arcface_torch/r50_features']

        # ds_features_r18 = [[], []]
        # ds_features_r50 = [[], []]
        ds_features_r18_interpolated = torch.empty(size=(0, 512))
        ds_features_r50_interpolated = torch.empty(size=(0, 512))
        ds_interpolated_lists = [ds_features_r18_interpolated, ds_features_r50_interpolated]

        label_list = []

        # ds_features_list = [ds_features_r18, ds_features_r50]
        # first_even_id = False
        #for i, batch in enumerate(dl):
        for [[prev_num_vectors, prev_label] , [num_vectors, label]]  in zip(zip(sorted, idx_sorted) ,zip(sorted[1:], idx_sorted[1:]) ):
            X1_prev, X2_prev = GetLabelVectors(prev_label, index, lengths, full_dataset)
            X1,X2 = GetLabelVectors(label,index,lengths,full_dataset)
            ds_features_r18 = [X1_prev,X1]
            ds_features_r50 = [X2_prev, X2]
            #X1, X2, y = batch
            #y = y.item()
            #if y > stop_label:
            #    break
            #if y % 2 == 0 and y != 0 and not first_even_id:
                #first_even_id = True
                # Take min size
            ds_features_list = [ds_features_r18, ds_features_r50]
            min_len = min([len(ds_features_list[0][0]), len(ds_features_list[0][1]),max_interp_num])
            max_len = max([len(ds_features_list[0][0]), len(ds_features_list[0][1])])
            if max_len<min_interp_num:
                # TODO complete
                break
            label_list += [current_label] * min_len
            current_label = current_label + 1

            for i_ds, ds_features in enumerate(ds_features_list):

                ds_features_label_even = ds_features[0][:min_len]
                ds_features_label_odd = ds_features[1][:min_len]
                ds_features_interpolated = (ds_features_label_even + ds_features_label_odd) / 2
                ds_interpolate = ds_interpolated_lists[i_ds]
                ds_interpolated_lists[i_ds] = torch.vstack((ds_interpolate, ds_features_interpolated))

                if ds_interpolated_lists[i_ds].shape[0] >= pkl_batch_size:
                    new_pkl_features = ds_interpolated_lists[i_ds][:pkl_batch_size, :]
                    idx_tensor = torch.tensor(list(range(n_images, n_images + pkl_batch_size)))
                    label_tensor = torch.tensor(label_list[:pkl_batch_size])

                    temp_tensor = torch.hstack((torch.unsqueeze(idx_tensor, dim=1),
                                                torch.unsqueeze(label_tensor, dim=1),
                                                new_pkl_features))

                    pkl_batch = os.path.join(paths[i_ds], str(pkl_file_i) + '.pkl')
                    with open(pkl_batch, 'wb') as file:
                        pickle.dump(temp_tensor, file)

                    ds_interpolated_lists[i_ds] = ds_interpolated_lists[i_ds][pkl_batch_size:, :]
                    if i_ds == 1:
                        print("Dumped ", pkl_batch)
                        label_list = label_list[pkl_batch_size:]
                        n_images = n_images + pkl_batch_size
                        pkl_file_i = pkl_file_i + 1
                        interpolated_cnt += pkl_batch_size

                # ds_features_r18 = [[], []]
                # ds_features_r50 = [[], []]
            #elif y % 2 != 0:
            #    first_even_id = False

            # ds_features_r18[y % 2].append(X1)
            # ds_features_r50[y % 2].append(X2)

    set.n_records = len(set) + (interpolated_cnt//interpolated_div)
    set.ds1.max_idx_offset = max_idx_offset
    set.ds2.max_idx_offset = max_idx_offset
    set.ds1.max_idx = max_idx
    set.ds2.max_idx = max_idx

    set_interpolated_data.n_records = interpolated_cnt // interpolated_div
    set_interpolated_data.ds1.max_idx_offset = len(full_dataset)
    set_interpolated_data.ds2.max_idx_offset = len(full_dataset)
    set_interpolated_data.ds1.max_idx = -1
    set_interpolated_data.ds2.max_idx = -1

    print("n_records", str(set.n_records), "interpolated count", str(interpolated_cnt), "Max Idx", str(max_idx))
    return set,set_interpolated_data

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
        print("Loading pkl dataset: " + pkl_dir_path)
        self.pkl_dir_path = pkl_dir_path
        with open(os.path.join(pkl_dir_path, 'metadata.pkl'), 'rb') as pkl_metadata:
            metadata_dict = pickle.load(pkl_metadata)
            self.n_records = metadata_dict["n_images"]
            self.classes = metadata_dict["classes"]
            self.dataset_batch_size = metadata_dict["batch_size"]
        self.files = [None] * (self.n_records // self.dataset_batch_size)
        self.max_idx_offset = 0
        self.index_offset = 0
        self.max_idx = -1
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

    def __getitem__(self, index: int) -> Tuple[Tensor, int, int]:
        index += self.index_offset

        if index > self.max_idx:
            index += self.max_idx_offset
        # i_batch = offset + index // self.dataset_batch_size
        i_batch = index // self.dataset_batch_size
        offset = index % self.dataset_batch_size
        batch_path = os.path.join(self.pkl_dir_path, str(i_batch) + '.pkl')
        # if self.files[i_batch] is None:
        # self.files[i_batch] = open(batch_path, 'rb')
        f = open(batch_path, 'rb')
        batch = pickle.load(f)
        idx = int(batch[offset, 0].item())
        label = int(batch[offset, 1].item())
        image = batch[offset, 2:]
        return image, label, idx
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
