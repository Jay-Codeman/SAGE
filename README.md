# CVPR 2025 Submission No.1926: Federated Semi-Supervised Learning via Pseudo-Correction Utilizing Confidence Discrepancy (SAGE)

## Setup

1. Create a new Python environment:
   ```bash
   conda create --name sage python=3.8.18 
   conda activate sage
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
Supported datasets:
* CIFAR-10
* CIFAR-100
* CINIC-10
* SVHN

Before training, please ensure the dataset paths are correctly set in `options.py`.

## Training
To reproduce the results presented in our paper, you can run the following command:
```bash
python SAGE.py --dataset='CIFAR100' --alpha=0.1 --gpu_id=0
```
Please replace `--dataset`, `--alpha`, and `--gpu_id` with appropriate values to customize the training configuration.
