import numpy as np
from torch.utils.data.dataset import Dataset
import copy
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC
from Dataset.sample_dirichlet import clients_indices, clients_indices_unlabel

import time



def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1



def show_clients_data_distribution(dataset, clients_indices_labeled, clients_indices_unlabeled, num_classes):
    dict_per_client_labeled = []

    for client, indices in enumerate(zip(clients_indices_labeled, clients_indices_unlabeled)):
        nums_data_labeled = [0 for _ in range(num_classes)]
        nums_data_unlabeled = [0 for _ in range(num_classes)]
        idx_labeled, idx_unlabeled = indices
        for idx in idx_labeled:
            label = dataset[idx][1]
            nums_data_labeled[label] += 1
        dict_per_client_labeled.append(nums_data_labeled)
        for idx in idx_unlabeled:
            label = dataset[idx][1]
            nums_data_unlabeled[label] += 1
        print(f'client {client} labeled number per class : {nums_data_labeled}')
        print(f'client {client} unlabeled number per class  : {nums_data_unlabeled}')
    return dict_per_client_labeled


def partition_train(list_label2indices: list, ipc):

    list_label2indices_labeled = []
    list_label2indices_unlabeled = []

    for indices in list_label2indices:

        idx_shuffle = np.random.permutation(indices)

        list_label2indices_labeled.append(idx_shuffle[:ipc])
        list_label2indices_unlabeled.append(idx_shuffle[ipc:])
    return list_label2indices_labeled, list_label2indices_unlabeled


def compute_clients_labeled_data_distribution(dataset, clients_indices_labeled, num_classes):
    dict_per_client_labeled = []
    nums_data_labeled = [0 for _ in range(num_classes)]
    for idx in clients_indices_labeled:
        label = dataset[idx][1]
        nums_data_labeled[label] += 1
    dict_per_client_labeled.append(nums_data_labeled)
    return dict_per_client_labeled


def partition_train_teach(list_label2indices: list, ipc, seed=None):
    random_state = np.random.RandomState(0)
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_teach.append(indices[:ipc])

    return list_label2indices_teach


def partition_unlabel(list_label2indices: list, num_data_train: int):
    random_state = np.random.RandomState(0)
    list_label2indices_unlabel = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_unlabel.append(indices[:num_data_train // 100])
    return list_label2indices_unlabel


def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res




class Indices2Dataset_labeled(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

        ##### fix self.dataset v1 ######
        self.client_dataset = [self.dataset[i] for i in indices]
        self.client_dataset *= 2000
        # 因为使用batch 128时，每次epoch都需要重新 iter(dataset) 一次，每次100ms
        # 这里复制多次dataset，减少运行 iter 函数的次数
        # 数字是随便定的

    def __getitem__(self, idx):
        self.label_trans = transforms.Compose([
       #     transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            ])

        ##### fix self.dataset v1 ######
        # idx = self.indices[idx]
        # image, label = self.dataset[idx]
        ##
        image, label = self.client_dataset[idx]

        image = self.label_trans(image)

        return image, label

    def __len__(self):
        # return len(self.indices)
        return len(self.client_dataset)



class Indices2Dataset_unlabeled_fixmatch(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

        ##### fix self.dataset v1 ######
        self.client_dataset = [self.dataset[i] for i in self.indices]
        self.client_dataset_len = len(self.client_dataset)
        self.client_dataset *= 50 # save time loading data


    def fixmatch(self, image):
        self.weak = transforms.Compose([
    #        transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),

            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
    #        transforms.ToTensor(),
        ])

        self.strong = transforms.Compose([
    #        transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),

            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            ])


        self.normalize = transforms.Compose([
    #        transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))])
        weak = self.weak(image)
        strong = self.strong(image)
        return self.normalize(weak), self.normalize(strong)

    def __getitem__(self, idx):

        ##### fix self.dataset v1 ######
        # idx = self.indices[idx]
        # image, label = self.dataset[idx]
        ##
        image, label = self.client_dataset[idx]

        image1, image2 = self.fixmatch(image)
        return image1, image2, label

    def __len__(self):
        ##### fix self.dataset v1 ######
        # return len(self.indices)
        # return len(self.client_dataset)
        return self.client_dataset_len




def list_IID_clients(list_label2indices_labeled, list_label2indices_unlabeled, num_classes, num_clients):

    labeled_per_client = int(len(list_label2indices_labeled[0]) / num_clients)
    unlabeled_per_client = int(len(list_label2indices_unlabeled[0]) / num_clients)

    random_state = np.random.RandomState(0)
    for class_id, data_all in enumerate(zip(list_label2indices_labeled, list_label2indices_unlabeled)):
        labeled_data, unlabeled_data = data_all
        list_label2indices_labeled[class_id] = [labeled_data[i:i + labeled_per_client] for i in range(0, len(labeled_data), labeled_per_client)]
        list_label2indices_unlabeled[class_id] = [unlabeled_data[i:i + unlabeled_per_client] for i in range(0, len(unlabeled_data), unlabeled_per_client)]

    list_clients_labeled = [[] for i in range(num_clients)]
    list_clients_unlabeled = [[] for i in range(num_clients)]
    for client_id in range(num_clients):
        for class_id in range(num_classes):
            list_clients_labeled[client_id].extend(list(list_label2indices_labeled[class_id][client_id]))
            list_clients_unlabeled[client_id].extend(list(list_label2indices_unlabeled[class_id][client_id]))

    return list_clients_labeled, list_clients_unlabeled




def sampling_labeled_data_non_iid(args, data_local_training, list_label2indices_labeled, num_labeled_client, alpha, seed=0):
    list_choose_labeled = []
    list_choose_labeled_client1 = []
    list_rest_label2indices_labeled = []
    random_state = np.random.RandomState(seed)


    list_choose_labeled_non_iid = clients_indices(list_label2indices=list_label2indices_labeled,
                                                  num_classes=args.num_classes,
                                                  num_clients=2,
                                                  non_iid_alpha=alpha,
                                                  seed=seed)


    client1_sampling = compute_clients_labeled_data_distribution(data_local_training,
                                                                 list_choose_labeled_non_iid[0],
                                                                 args.num_classes)
    client1_sampling = client1_sampling[0]
    for class_idx, list_index in enumerate(list_label2indices_labeled):

        new_data = set(random_state.choice(list_index, client1_sampling[class_idx], replace=False))
        list_new_data = list(new_data)
        list_choose_labeled_client1.extend(list_new_data)
        list_index = list(set(list_index) - new_data)
        list_rest_label2indices_labeled.append(list_index)

    list_choose_labeled.append(list_choose_labeled_client1)


    list_choose_labeled_rest_client = clients_indices_unlabel(list_label2indices=list_rest_label2indices_labeled,
                                                        num_classes=args.num_classes,
                                                        num_clients=(num_labeled_client-1),
                                                        non_iid_alpha=alpha,
                                                        seed=10)
    list_choose_labeled.extend(list_choose_labeled_rest_client)

    return list_choose_labeled


def sampling_unlabeled_data_non_iid(args, list_label2indices_unlabeled,
                                    num_unlabeled_client, alpha, seed=0):
    list_choose_unlabeled = []
    list_unlabeled_part1 = []
    list_unlabeled_part2 = []
    random_state = np.random.RandomState(0)
    class_sampling = [2000] * 10
    for class_idx, list_index in enumerate(list_label2indices_unlabeled):
        new_data = set(random_state.choice(list_index, class_sampling[class_idx], replace=False))
        list_new_data = list(new_data)
        list_unlabeled_part1.append(list_new_data)
        list_index = list(set(list_index) - new_data)
        list_unlabeled_part2.append(list_index)


    list_client_part1 =clients_indices_unlabel(list_label2indices=list_unlabeled_part1,
                                        num_classes=args.num_classes, num_clients=9,
                                        non_iid_alpha=alpha, seed=1000)
    list_client_part2 = clients_indices_unlabel(list_label2indices=list_unlabeled_part2, num_classes=args.num_classes, num_clients=10,
                                        non_iid_alpha=alpha, seed=1000)
    list_choose_unlabeled.append([])
    list_choose_unlabeled.extend(list_client_part1)
    list_choose_unlabeled.extend(list_client_part2)

    return list_choose_unlabeled

