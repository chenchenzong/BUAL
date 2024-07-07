
## 1. Requirements
### Environments
Our experiments are all performed on a single GPU 2080 and require following packages.

- CUDA 10.1
- python == 3.7.10
- pytorch == 1.7.1
- torchvision == 0.8.2
- numpy == 1.20.2

### Datasets 
For CIFAR10 or CIFAR100, please download it to `~/data`.
For TinyImagenet, please download it to `~/tiny-imagenet-200`.

## 2. Training

```train
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --init_ratio 0.01 --seed 1 --total_times 8 --aux_epochs 100 --pl_epochs 100 --nl_epochs 100 --known_classes 4 --query_size 1500
```
* --dataset: cifar10, cifar100, tinyimagenet.
* --init_ratio: the proportion of the initial labeled data. In our experiment, we set 0.01 for cifar 10, 0.08 for cifar100 and tinyimagenet.
* --seed: 1, 2, 3, etc.
* --total_times: the total query number. In our experiment, we set --total_times 8.
* --aux_epochs: the total training epoch of the Auxiliary classifier. In our experiment, we set --aux_epochs 100.
* --pl_epochs: the total training epoch of the positive learning classifier (target model). In our experiment, we set --pl_epochs 100.
* --nl_epochs: the total training epoch of the negative learning classifier. In our experiment, we set --nl_epochs 100.
* --known_classes: the total number of known classes. In our experiment, we set {2, 4, 6, 8} for cifar 10, {20, 40, 60, 80} for cifar100 and {40, 80, 120, 160} for tinyimagenet.
* --query_size: The total number of samples in a query. In our experiment, we set 1500 for cifar 10 and cifar100, 3000 for tinyimagenet.
