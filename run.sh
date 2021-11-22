# CIFAR-10

## Baseline FixMatch
CUDA_VISIBLE_DEVICES= python fixmatch.py --filters=32 --dataset=cifar10_LT_20.1@20-1 \
 --train_dir ./experiments/fixmatch 

CUDA_VISIBLE_DEVICES= python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@50-1 \
 --train_dir ./experiments/fixmatch 

CUDA_VISIBLE_DEVICES= python fixmatch.py --filters=32 --dataset=cifar10_LT_100.1@100-1 \
 --train_dir ./experiments/fixmatch 

## Ours
CUDA_VISIBLE_DEVICES= python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@50-1 \
--CAP --CAI --CADR --train_dir ./experiments/fixmatch 

CUDA_VISIBLE_DEVICES= python fixmatch.py --filters=32 --dataset=cifar10_LT_50.1@50-1 \
--CAP --CAI --CADR --train_dir ./experiments/fixmatch 

CUDA_VISIBLE_DEVICES= python fixmatch.py --filters=32 --dataset=cifar10_LT_100.1@100-1 \
--CAP --CAI --CADR --train_dir ./experiments/fixmatch 


# CIFAR-100
## Baseline FixMatch
CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar50_LT_50.1@50-1 \
 --train_dir ./experiments/fixmatch 

CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar100_LT_100.1@100-1 \
 --train_dir ./experiments/fixmatch 

CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=128 --wd=0.001 --dataset=cifar100_LT_200.1@200-1 \
 --train_dir ./experiments/fixmatch 

## Ours
CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=128 --wd=0.0015 --dataset=cifar50_LT_50.1@50-1 \
--CAP --CAI --CADR --train_dir ./experiments/fixmatch 

CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=128 --wd=0.0015 --dataset=cifar100_LT_100.1@100-1 \
--CAP --CAI --CADR --train_dir ./experiments/fixmatch 

CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=128 --wd=0.0015 --dataset=cifar100_LT_200.1@200-1 \
--CAP --CAI --CADR --train_dir ./experiments/fixmatch 


# stl-10
## Baseline FixMatch
CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=32 --scales=4 --dataset=stl10_LT_50.1@50-1 \
 --train_dir ./experiments/fixmatch 

CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=32 --scales=4 --dataset=stl10_LT_100.1@100-1 \
 --train_dir ./experiments/fixmatch 

## Ours
CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=32 --scales=4 --dataset=stl10_LT_50.1@50-1 \
--CAP --CAI --CADR --train_dir ./experiments/fixmatch 

CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=32 --scales=4 --dataset=stl10_LT_100.1@100-1 \
--CAP --CAI --CADR --train_dir ./experiments/fixmatch 


# miniImageNet
## Baseline FixMatch
CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=64 --dataset=miniImageNet_LT_100.1@100-1 \
 --train_dir ./experiments/fixmatch 

## Ours
CUDA_VISIBLE_DEVICES=  python fixmatch.py --filters=64 --dataset=miniImageNet_LT_100.1@100-1 \
--CAP --CAI --CADR --train_dir ./experiments/fixmatch 

