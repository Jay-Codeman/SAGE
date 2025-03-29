# SAGE

<div align="center">

<!-- Center-No-Ver-Bar-H1-H2 for GitHub, fork from https://gist.github.com/CodeByAidan/bb43bdb1c07c0933d8b67c23515fb912 -->
<div id="toc">
	<ul align="center" style="list-style: none">
		<summary>
			<h2> Mind the Gap: Confidence Discrepancy Can Guide Federated Semi-Supervised Learning
            Across Pseudo-Mismatch </h2>
			<h3> —— CVPR 2025 —— </h3>
		</summary>
	</ul>
</div>



[![arXiv](https://img.shields.io/badge/arXiv-2503.13227-b31b1b.svg)](https://arxiv.org/abs/2503.13227)
[![cvpr](https://img.shields.io/badge/CVPR-HomePage-blue)](https://cvpr.thecvf.com/virtual/2025/poster/33062)

*Original Title: Federated Semi-Supervised Learning via Pseudo-Correction utilizing Confidence Discrepancy.*

![pipeline.png](pipeline.png)

</div>



### Abstract
Federated Semi-Supervised Learning (FSSL) aims to leverage unlabeled data across clients with limited labeled data to train a global model with strong generalization ability. Most FSSL methods rely on consistency regularization with pseudo-labels, converting predictions from local or global models into hard pseudo-labels as supervisory signals. However, we discover that the quality of pseudo-label is largely deteriorated by data heterogeneity, an intrinsic facet of federated learning. In this paper, we study the problem of FSSL in-depth and show that (1) heterogeneity exacerbates pseudo-label mismatches, further degrading model performance and convergence, and (2) local and global models' predictive tendencies diverge as heterogeneity increases. Motivated by these findings, we propose a simple and effective method called **S**emi-supervised **A**ggregation for **G**lobally-Enhanced **E**nsemble (SAGE), that can flexibly correct pseudo-labels based on confidence discrepancies. This strategy effectively mitigates performance degradation caused by incorrect pseudo-labels and enhances consensus between local and global models. Experimental results demonstrate that SAGE outperforms existing FSSL methods in both performance and convergence.

### Setup

1. Create a new Python environment:

   ```bash
   conda create --name sage python=3.8.18 
   conda activate sage
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Dataset

Supported datasets:

* CIFAR-10
* CIFAR-100
* CINIC-10
* SVHN

Before running, please ensure the dataset paths are correctly set in `options.py`.

### Usage

Here is an example shell script to run SAGE on CIFAR-100 :

```bash
python SAGE.py --dataset='CIFAR100' --alpha=0.1 --gpu_id=0
```

Please replace `--dataset`, `--alpha`, and `--gpu_id` with appropriate values to customize the training configuration.



