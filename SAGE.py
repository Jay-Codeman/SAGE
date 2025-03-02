from torchvision import datasets
from torchvision.transforms import transforms
from options import args_parser
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset_labeled, \
    Indices2Dataset_unlabeled_fixmatch, partition_train
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
import numpy as np
import pandas as pd
from torch import max, eq, no_grad
from Dataset.CINIC10 import CINIC10
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from Model.resnet import ResNet
from tqdm import tqdm
import copy
import torch
import random
from torch.utils.data import DataLoader, RandomSampler
import logging
import os

worker_num = 4


class Global(object):
    def __init__(self, args):
        self.model = ResNet(resnet_size=8, scaling=4,
                            save_activations=False, group_norm_num_groups=None,
                            freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)

        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes


    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # FedAvg
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.cuda(args.gpu_id), labels.cuda(args.gpu_id)
                _, outputs = self.model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.model.state_dict()


class Local(object):
    def __init__(self, args):

        self.local_model = ResNet(resnet_size=8, scaling=4,
                                  save_activations=False, group_norm_num_groups=None,
                                  freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)


        self.local_G = ResNet(resnet_size=8, scaling=4,
                              save_activations=False, group_norm_num_groups=None,
                              freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)

        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)

        self.criterion = CrossEntropyLoss().cuda(args.gpu_id)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)


    def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r):

        self.labeled_trainloader = DataLoader(
            dataset=data_client_labeled,
            sampler=RandomSampler(data_client_labeled),
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True,
            num_workers = 2,
            pin_memory=True
        )

        self.unlabeled_trainloader = DataLoader(
            dataset=data_client_unlabeled,
            sampler=RandomSampler(data_client_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True,
            num_workers=2,
            pin_memory=True
        )

        self.local_model.load_state_dict(global_params)
        self.local_model.train()

        self.local_G.load_state_dict(global_params)
        self.local_G.eval()


        # default: E = 5
        for local_epoch in range(args.local_epochs):

            labeled_iter = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)

            local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

            for epoch in range(local_iter):

                try:
                    inputs_x, targets_x = labeled_iter.__next__()
                except:
                    labeled_iter = iter(self.labeled_trainloader)
                    inputs_x, targets_x = labeled_iter.__next__()

                try:
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()
                except:
                    unlabeled_iter = iter(self.unlabeled_trainloader)
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()

                inputs_x, inputs_u_w, inputs_u_s = inputs_x.cuda(args.gpu_id), inputs_u_w.cuda(args.gpu_id), inputs_u_s.cuda(args.gpu_id)

                batch_size = inputs_x.shape[0]
                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id) # 将有标签和无标签数据交错组合成一个大的批次
                targets_x = targets_x.cuda(args.gpu_id)

                # local logits
                _, logits = self.local_model(inputs)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')


                # global pseudo-labeling
                _, logits_u_w_global = self.local_G(inputs_u_w.cuda(args.gpu_id))
                pseudo_label_global = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
                max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)

                # local pseudo-labeling
                pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)

                targets_u_local_one_hot = F.one_hot(targets_u_local, args.num_classes).float()
                targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()



                mask_local = max_probs_local.ge(args.threshold).float()
                mask_global = max_probs_global.ge(args.threshold).float()

                # delta_p
                delta_c = torch.abs(max_probs_local - max_probs_global) + 1e-6
                delta_c = torch.clamp(delta_c, min=1e-6, max=1.0)

                # Confidence-Driven Soft Correction
                # Set sensitivity parameter kappa
                # Ensure lambda_dynamic = 0.5 when delta_p = 0.05
                # kappa = log(2) / 0.05 based on the formula lambda_dynamic = exp(-kappa * delta_p)
                kappa = torch.log(torch.tensor(2.0)) / 0.05  # calculate kappa
                lambda_dynamic = torch.exp(-kappa * delta_c)
                lambda_dynamic = torch.clamp(lambda_dynamic, min=1e-6, max=1.0)

                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot + (1-lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,  # 使用动态伪标签修正
                    targets_u_global_one_hot
                )

                mask_valid = torch.max(mask_local, mask_global)

                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1)

                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u = final_targets_u + 1e-10
                Lu = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid).mean()


                loss = Lx + args.lambda_u * Lu

                # logging
                logging.info(
                    f'Round {r}, Local Epoch {local_epoch}, Batch {epoch}: Lx = {Lx.item():.4f}, Lu = {Lu.item():.4f}, lambda_dynamic = {lambda_dynamic[0].item():.4f}')


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return copy.deepcopy(self.local_model.state_dict())


    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])




def fixmatch(alpha):

    args = args_parser()

    log_dir = f'./results/{args.dataset}/logs'
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'SAGE α={alpha}.log'.format(alpha = alpha))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file
                        )

    if args.dataset == 'CIFAR10':
        args.num_classes = 10
        args.num_labeled = 500
        args.num_rounds = 300
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)

    elif args.dataset == 'CIFAR100':
        args.num_classes = 100
        args.num_labeled = 50
        args.num_rounds = 500
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_test)


    elif args.dataset == 'SVHN':
        args.num_classes = 10
        args.num_labeled = 460
        args.num_rounds = 150
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        data_local_training = datasets.SVHN(args.path_svhn, split='train', download=True, transform=None)
        data_global_test = datasets.SVHN(args.path_svhn, split='test', transform=transform_test, download=True)

    elif args.dataset == 'CINIC10':
        args.num_classes = 10
        args.num_labeled = 900
        args.num_rounds = 400
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
        ])
        data_local_training = CINIC10(root=args.path_cinic10, split='train', transform=None)
        data_global_test = CINIC10(root=args.path_cinic10, split='test', transform=transform_test)

    else:
        print(
            f"Error: Unsupported dataset {args.dataset}. Please specify one of the following: CIFAR10, CIFAR100, CINIC10 or SVHN.")
        exit(1)

    print(
        'dataset:{dataset}\n'
        'num_classes:{num_classes}\n'
        'num_labeled:{num_labeled}\n'
        'non_iid:{alpha}\n'
        'mu:{mu}\n'
        'num_rounds:{num_rounds}\n'
        'batch_label:{batch_label}, batch_unlabel:{batch_unlabel}'.format(
            dataset=args.dataset,
            num_classes=args.num_classes,
            num_labeled=args.num_labeled,
            alpha=alpha,
            mu=args.mu,
            num_rounds=args.num_rounds,
            batch_label=args.batch_size_local_labeled,
            batch_unlabel=args.batch_size_local_unlabeled,
        ))

    random_state = np.random.RandomState(args.seed)

    list_label2indices = classify_label(data_local_training, args.num_classes)

    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, args.num_labeled)

    # IID
    if alpha == 0:
        list_client2indices_labeled = clients_indices_homo(list_label2indices=list_label2indices_labeled,
                                                           num_classes=args.num_classes,
                                                           num_clients=args.num_clients)
        list_client2indices_unlabeled = clients_indices_homo(list_label2indices=list_label2indices_unlabeled,
                                                             num_classes=args.num_classes,
                                                             num_clients=args.num_clients)
    # Non-IID
    else:
        list_client2indices_labeled = clients_indices(list_label2indices=list_label2indices_labeled,
                                                      num_classes=args.num_classes,
                                                      num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)
        list_client2indices_unlabeled = clients_indices(list_label2indices=list_label2indices_unlabeled,
                                                        num_classes=args.num_classes,
                                                        num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)

    # Show data distribubution
    show_clients_data_distribution(data_local_training, list_client2indices_labeled,
                                   list_client2indices_unlabeled, args.num_classes)

    # add labeled samples without labels into the unlabeled dataset
    for client in range(args.num_clients):
        list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])

    global_model = Global(args)
    local_model = Local(args)

    total_clients = list(range(args.num_clients))


    indices2data_labeled = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)

    fedavg_acc = []

    # FL training
    for r in tqdm(range(1, args.num_rounds + 1), desc='Server'):


        dict_global_params = global_model.download_params()

        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []

        # client training
        for client in online_clients:
            indices2data_labeled.load(list_client2indices_labeled[client])
            data_client_labeled = indices2data_labeled
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            data_client_unlabeled = indices2data_unlabeled


            list_nums_local_data.append(len(data_client_labeled) + len(data_client_unlabeled))
            local_params = local_model.fixmatch_train(args, data_client_labeled, data_client_unlabeled,
                                                      copy.deepcopy(dict_global_params), r)
            list_dicts_local_params.append(copy.deepcopy(local_params))


        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)

        global_acc = global_model.fedavg_eval(copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test)
        fedavg_acc.append(global_acc)
        print('round {round} accuracy:{global_acc}'.format(
            round = r,
            global_acc = global_acc))


        result_dir = f'./results/{args.dataset}'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/SAGE α={alpha}.csv'
        acc_num_pseudo_label_csv_col_name = ['acc', 'num_pseudo_label']
        acc_num_pseudo_label_csv_index = list(range(1, len(fedavg_acc)+1))
        acc_num_pseudo_label_csv_df = pd.DataFrame({'acc':fedavg_acc}, index = acc_num_pseudo_label_csv_index)
        # 保存文件
        acc_num_pseudo_label_csv_df.to_csv(result_file, encoding='utf8')






if __name__ == '__main__':

    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)
    torch.backends.cudnn.deterministic = True  # cudnn

    args = args_parser()
    fixmatch(args.alpha)
