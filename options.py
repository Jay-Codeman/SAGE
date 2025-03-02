# -*-coding:utf-8-*-
import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        help='CIFAR10 / CIFAR100 / SVHN / CINIC10')
    parser.add_argument('--num_clients', type=int, default=20,
                        help='Total number of clients in the federated learning setup')
    parser.add_argument('--num_online_clients', type=int, default=8,
                        help='Number of clients participating in each communication round')
    parser.add_argument('--mu', default=2, type=int,
                        help='Number of augmentations for unlabeled samples')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Dirichlet distribution')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='Pseudo-label threshold')
    parser.add_argument('--lambda_u', default=1, type=float,
                        help='Coefficient of unlabeled loss')
    parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=128)
    parser.add_argument('--kappa', default=0.5, type=float,
                        help='Hyperparameter for controlling sensitivity to confidence discrepancy in CDSC')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='local training epochs in local training')
    parser.add_argument('--batch_size_local_labeled', type=int, default=128)
    parser.add_argument('--batch_size_local_unlabeled', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=512)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--lr_distillation_training', type=float, default=0.1)

    # dataset path
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--path_svhn', type=str, default=os.path.join(path_dir, 'data/SVHN/'))
    parser.add_argument('--path_cinic10', type=str, default=os.path.join(path_dir, 'data/CINIC10/'))


    #------------- ablation study -------------#
    parser.add_argument('--SAGE_fixed_p', default=0.5, type=float,
                        help='Fixed parameter used in SAGE ablation study instead of dynamic adjustment')
    parser.add_argument('--ablation_kappa', default=2, type=float,
                        help='Hyperparameter for controlling sensitivity of exponential decay in SAGE ablation study')

    parser.add_argument('--seed', type=int, default=7)


    # ------------- baselines -------------#
    # FedProx
    parser.add_argument('--lambda_prox', default=0.001, type=float,
                        help='coefficient of FedProx')
    # FedLabel
    parser.add_argument('--fedlabel_lambda', default=1, type=float,
                        help='FedLabel Hyperparameter')

    # FreeMatch
    parser.add_argument('--freematch_sat_ema', default=0.999, type=float,
                        help='ema weight for FreeMatch SAT')
    parser.add_argument('--sat_loss_ratio', default=1, type=float,
                        help='Weight for FreeMatch SAT loss')
    parser.add_argument('--saf_loss_ratio', default=0.05, type=float,
                        help='Weight for FreeMatch SAF loss')

    # FedMatch
    parser.add_argument('--fedmatch_confidence_threshold', type=float, default=0.75)
    parser.add_argument('--fedmatch_lambda_s', default=10, type=float)
    parser.add_argument('--fedmatch_lambda_iccs', default=0.01, type=float)
    parser.add_argument('--fedmatch_lambda_l1', default=0.0001, type=float)
    parser.add_argument('--fedmatch_lambda_l2', default=10, type=float)  # 10
    parser.add_argument('--fedmatch_H', default=2, type=int)
    parser.add_argument('--T', default=1, type=float,
                        help='Temperature of pseudo-labeling')

    parser.add_argument('--num_epochs_label_distillation', type=int, default=50)
    parser.add_argument('--num_epochs_unlabel_distillation', type=int, default=50)
    parser.add_argument('--batch_size_label_distillation', type=int, default=128)
    parser.add_argument('--batch_size_unlabel_distillation', type=int, default=128)


    args = parser.parse_args()
    return args

